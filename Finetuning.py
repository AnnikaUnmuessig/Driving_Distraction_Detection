"""
Finetuning.py
Fine-tunes TimeSformer-HR (16 frames, 448x448) on the distraction dataset.

Dataset: endoard/distraction_dataset (downloaded locally via download_assets.py)
Model:   facebook/timesformer-hr-finetuned-k400

Metrics: Top-1 Accuracy, Top-3 Accuracy, Weighted F1

Environment variables (set in sbatch or locally):
  DATASET_PATH  — path to the local dataset folder (default: ./distraction_dataset)
  MODEL_PATH    — path to the local model folder   (default: facebook/timesformer-hr-finetuned-k400)
  OUTPUT_DIR    — where to save checkpoints         (default: ./timesformer-hr-16)
"""

import os
import random
import cv2
import json
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoImageProcessor,
    TimesformerForVideoClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    TrainerCallback,
)

class CustomLoggingCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"\n[{state.epoch + 1:.0f}/{args.num_train_epochs}] >>> Inizio Epoca...")
        
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"[{state.epoch:.0f}/{args.num_train_epochs}] <<< Epoca Terminata.")
        
    def on_save(self, args, state, control, **kwargs):
        print(f"Checkpoint salvato allo step {state.global_step}.")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            print(f"Metriche: {logs}")

# ── Configuration ──────────────────────────────────────────────────────────────

# Resolved from env vars so the sbatch script can override paths without editing code
MODEL_NAME   = os.environ.get("MODEL_PATH",   "facebook/timesformer-hr-finetuned-k400")
DATASET_PATH = os.environ.get("DATASET_PATH", "./distraction_dataset")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",   "./timesformer-hr-16")

NUM_FRAMES = 16     # TimeSformer-HR native frame count
LIMIT_CAP  = 160    # Max clips per class — keeps training balanced while new data is added
                    # Set to None to disable capping
SEED       = 42

# 11 classes consistent with Data_preparation.py
# (reach_backseat, stand_still_waiting, unclassified are excluded intentionally)
CLASS_MAP = {
    "safe_driving":        0,
    "texting_right":       1,
    "phonecall_right":     2,
    "texting_left":        3,
    "phonecall_left":      4,
    "radio":               5,
    "drinking":            6,
    "reach_side":          7,
    "hair_and_makeup":     8,
    "talking_to_passenger": 9,
    "change_gear":         10,
}
ID2LABEL = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

# ── Dataset utilities ──────────────────────────────────────────────────────────

def build_entries(dataset_path, class_map, limit_cap=None, seed=42):
    """
    Scan dataset_path/<class_name>/ for video files.
    Optionally cap each class at limit_cap samples (shortest-first not needed
    here since we shuffle before capping).
    Returns list of (absolute_path, label_int).
    """
    rng = random.Random(seed)
    entries = []
    for class_name, label in class_map.items():
        class_dir = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Class folder not found, skipping: {class_dir}")
            continue
        videos = [
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".mp4", ".avi", ".mov"))
        ]
        rng.shuffle(videos)
        if limit_cap and len(videos) > limit_cap:
            videos = videos[:limit_cap]
        for v in videos:
            entries.append((v, label))
        print(f"  {class_name:25s}: {len(videos):4d} clips")
    return entries


def stratified_split(entries, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Stratified split so every class keeps the same proportion in each split.
    Returns (train, val, test) lists of (path, label) tuples.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for path, label in entries:
        by_class[label].append((path, label))

    train, val, test = [], [], []
    for label, items in by_class.items():
        rng.shuffle(items)
        n       = len(items)
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        train.extend(items[:n_train])
        val.extend(items[n_train: n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    return train, val, test

# ── Dataset class ──────────────────────────────────────────────────────────────

class VideoDataset(Dataset):
    def __init__(self, entries, processor, num_frames=16):
        self.entries    = entries
        self.processor  = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        video_path, label = self.entries[idx]
        frames = self._sample_frames(video_path)
        inputs = self.processor(images=frames, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }

    def _sample_frames(self, video_path):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.num_frames  # fallback for corrupt/empty videos

        indices = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        # Pad if any reads failed (duplicate last frame)
        if not frames:
            frames = [Image.new("RGB", (448, 448))] * self.num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        return frames[:self.num_frames]

# ── Metrics ────────────────────────────────────────────────────────────────────

_accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    # Top-1 accuracy
    acc = _accuracy_metric.compute(
        predictions=predictions, references=labels
    )["accuracy"]

    # Top-3 accuracy
    top3_indices = np.argsort(logits, axis=-1)[:, -3:]
    top3_acc = float(np.mean([labels[i] in top3_indices[i] for i in range(len(labels))]))

    # Weighted F1 (handles class imbalance in the eval set)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy":      round(acc,      4),
        "top3_accuracy": round(top3_acc, 4),
        "f1_weighted":   round(f1,       4),
    }

# ── Custom Trainer with WeightedRandomSampler ──────────────────────────────────

class BalancedTrainer(Trainer):
    """
    Overrides get_train_dataloader to use WeightedRandomSampler,
    giving rare classes the same expected frequency as common ones.
    """
    def get_train_dataloader(self):
        labels        = [item[1] for item in self.train_dataset.entries]
        class_counts  = Counter(labels)
        class_weight  = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = torch.tensor(
            [class_weight[l] for l in labels], dtype=torch.float
        )
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=default_data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
        )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Model:      {MODEL_NAME}")
    print(f"Dataset:    {DATASET_PATH}")
    print(f"Output:     {OUTPUT_DIR}")
    print(f"Frames:     {NUM_FRAMES}")
    print(f"Limit cap:  {LIMIT_CAP}")
    print()

    # ── Load processor & model ────────────────────────────────────────────────
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=CLASS_MAP,
        ignore_mismatched_sizes=True,  # replaces the K400 classification head
    )

    # ── Build splits ──────────────────────────────────────────────────────────
    print("Scanning dataset...")
    entries = build_entries(DATASET_PATH, CLASS_MAP, limit_cap=LIMIT_CAP, seed=SEED)
    if not entries:
        raise RuntimeError(f"No video files found under {DATASET_PATH}. "
                           "Did you run download_assets.py first?")

    train_entries, val_entries, test_entries = stratified_split(entries, train_ratio=0.7, val_ratio=0.2, seed=SEED)
    print(f"\nSplit sizes — Train: {len(train_entries)} | "
          f"Val: {len(val_entries)} | Test: {len(test_entries)}")

    label_dist = Counter([e[1] for e in train_entries])
    print("Train class distribution:")
    for lbl, cnt in sorted(label_dist.items()):
        print(f"  {ID2LABEL[lbl]:25s}: {cnt}")

    train_dataset = VideoDataset(train_entries, processor, num_frames=NUM_FRAMES)
    val_dataset   = VideoDataset(val_entries,   processor, num_frames=NUM_FRAMES)
    test_dataset  = VideoDataset(test_entries,  processor, num_frames=NUM_FRAMES)

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        remove_unused_columns=False,

        # Batch / accumulation: effective batch = 4 * 4 = 16 per GPU
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        fp16=True,

        # Optimizer & scheduler
        learning_rate=2e-5,
        num_train_epochs=10,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        weight_decay=0.01,

        # Logging & evaluation
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,

        dataloader_num_workers=2,
        disable_tqdm=True,

        # W&B — set WANDB_MODE=offline in sbatch if no internet on compute node
        report_to="wandb",
        run_name="timesformer-hr-16frames",
        seed=SEED,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5), CustomLoggingCallback()],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\nStarting training...")
    trainer.train()

    # ── Final evaluation on test set ──────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    print("Test metrics:")
    for k, v in test_results.metrics.items():
        print(f"  {k}: {v}")

    # Generate Confusion Matrix
    print("\nGenerating confusion matrix...")
    preds = np.argmax(test_results.predictions, axis=1)
    labels = test_results.label_ids
    cm = confusion_matrix(labels, preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)],
                yticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save training history
    log_history = trainer.state.log_history
    log_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=4)
    print(f"Training metrics history saved to {log_path}")

    # ── Save best model ───────────────────────────────────────────────────────
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"\nBest model saved to: {best_model_dir}")


if __name__ == "__main__":
    main()