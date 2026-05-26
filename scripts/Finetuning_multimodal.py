#Attempt 2 if firs one fails
#-> Using Gemma-4 a multimodal LLM

import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import Trainer, default_data_collator

MODEL_NAME = "google/gemma-4-e4b-it"
NUM_FRAMES = 8
OUTPUT_DIR = "./gemma4-distraction"

LABELS = [
    "safe driving", "texting right hand", "phone call right hand",
    "texting left hand", "phone call left hand", "adjusting radio",
    "drinking", "reaching behind", "hair or makeup",
    "talking to passenger", "other distraction",
]

SYSTEM_PROMPT = (
    "You are a driving safety analyst. Classify the driver's activity into one of:\n"
    + "\n".join(f"- {l}" for l in LABELS)
    + "\nRespond with only the category name."
)


class VideoDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.df = pd.read_csv(csv_file, sep=" ", names=["path", "label"])
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path  = self.df.iloc[idx]["path"]
        label = LABELS[int(self.df.iloc[idx]["label"])]
        frames = self._sample_frames(path)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                + [{"type": "text", "text": "What is the driver doing?"}]},
            {"role": "assistant", "content": label},
        ]

        text   = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=text, images=[frames], return_tensors="pt",
                                max_length=2048, truncation=True, padding="max_length")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    def _sample_frames(self, path):
        cap    = cv2.VideoCapture(path)
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in np.linspace(0, total - 1, NUM_FRAMES, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            raise ValueError(f"No frames from {path}")
        while len(frames) < NUM_FRAMES:   # pad if any reads failed
            frames.append(frames[-1])
        return frames

# ── Model ──────────────────────────────────────────────────────────────────────

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear",
                    "gate_proj.linear", "up_proj.linear", "down_proj.linear"],
))
model.print_trainable_parameters()

Trainer(
    model=model,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        report_to="wandb",
        run_name="gemma4-e4b-distraction",
    ),
    train_dataset=VideoDataset("DMD_dataset/train.csv", processor),
    eval_dataset=VideoDataset("DMD_dataset/val.csv", processor),
    data_collator=default_data_collator,
).train()

model.save_pretrained(f"{OUTPUT_DIR}/adapter")
processor.save_pretrained(f"{OUTPUT_DIR}/adapter")