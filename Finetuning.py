import cv2
import numpy as np
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from PIL import Image
from transformers import TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import Trainer, default_data_collator
import os
import evaluate

metric = evaluate.load("accuracy")

#Fixed variables
num_frames = 96

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

class DrivingDistractionDataset(Dataset):
    def __init__(self, csv_file, processor, num_frames=96):
        self.df = pd.read_csv(csv_file, sep=' ', names=['path', 'label'])
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def sample_frames(self, video_path, num_frames):
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate indices to jump to
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR (OpenCV) to RGB (PIL/Transformers)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame))
            
            cap.release()
            return frames

    def __getitem__(self, idx):
        #Use path directly from CSV
        video_path = self.df.iloc[idx]['path']
        label = int(self.df.iloc[idx]['label'])
        
        #Extract frames
        frames = self.sample_frames(video_path, self.num_frames)
        
        #Process frames into the format TimeSformer expects Channels, Time, Height, Width)
        inputs = self.processor(images=frames, return_tensors="pt")
        
        return {
            "pixel_values": inputs['pixel_values'].squeeze(0), 
            "labels": torch.tensor(label)
        }

model_name = "fcakyon/timesformer-large-finetuned-k400" #timesformer pretrained on Kinetics-400 dataset version large
processor = AutoImageProcessor.from_pretrained(model_name)
model = TimesformerForVideoClassification.from_pretrained(model_name, num_labels=11, ignore_mismatched_sizes=True)

train_dataset = DrivingDistractionDataset(csv_file="DMD_dataset/train.csv", processor=processor, num_frames=num_frames)
val_dataset = DrivingDistractionDataset(csv_file="DMD_dataset/val.csv", processor=processor, num_frames=num_frames)
test_dataset = DrivingDistractionDataset(csv_file="DMD_dataset/test.csv", processor=processor, num_frames=num_frames)

sample = train_dataset[0]
print("Shape:", sample['pixel_values'].shape)       # expect [96, 3, 224, 224] or similar
print("Label:", sample['labels'])                   # expect 0-10
print("NaN?", torch.isnan(sample['pixel_values']).any())  # expect False
print("Min/Max:", sample['pixel_values'].min().item(), sample['pixel_values'].max().item())

training_args = TrainingArguments(
    output_dir="./timesformer-L-96",
    remove_unused_columns=False,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=16,    
    gradient_checkpointing=True,        
    fp16=False,                        
    learning_rate=2e-5,                 
    num_train_epochs=5,
    logging_steps=1,

    report_to="wandb",
    run_name="timesformer-large-96-frames", 
    eval_strategy="steps",            
    eval_steps=50,                         
    logging_first_step=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=default_data_collator, # Crucial for stacking pixel_values
)

trainer.train()