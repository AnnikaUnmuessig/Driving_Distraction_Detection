"""
videomae_finetuning.py
Fine-tunes VideoMAE on the driving distraction dataset.

Why VideoMAE over TimeSformer for small datasets:
  - Pre-trained with Masked Autoencoder (90-95% tube masking) → forces deep
    spatiotemporal understanding → far more data-efficient fine-tuning.
  - Kinetics-400 checkpoint: MCG-NJU/videomae-base-finetuned-kinetics
  - SSv2 checkpoint (hand actions, closer to driving): MCG-NJU/videomae-base-finetuned-ssv2

Environment variables (override from sbatch / notebook):
  MODEL_PATH      — HF model ID or local path   (default: MCG-NJU/videomae-base-finetuned-kinetics)
  DATASET_PATH    — local dataset folder         (default: ./distraction_dataset)
  OUTPUT_DIR      — checkpoint / stats output    (default: ./videomae_outputs)
  TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, GRAD_ACCUM_STEPS  — memory tuning
  HF_REPO_ID      — HF Hub repo for checkpoint push (leave empty to skip)
"""

import os
import random
import json
import cv2
import numpy as np
import torch
import evaluate
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter, defaultdict
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ── Callback ───────────────────────────────────────────────────────────────────

class CustomLoggingCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch) + 1
        print(f"\n{'='*60}")
        print(f"  Epoca {epoch_num}/{int(args.num_train_epochs)} — step globale: {state.global_step}")
        print(f"{'='*60}")

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch)
        metrics = {}
        for entry in reversed(state.log_history):
            if "loss" in entry or "eval_loss" in entry:
                metrics = entry
                break
        parts = []
        if "loss"          in metrics: parts.append(f"loss={metrics['loss']:.4f}")
        if "eval_loss"     in metrics: parts.append(f"eval_loss={metrics['eval_loss']:.4f}")
        if "eval_accuracy" in metrics: parts.append(f"acc={metrics['eval_accuracy']:.4f}")
        print(f"  [OK] Epoca {epoch_num}/{int(args.num_train_epochs)} completata  |  {' | '.join(parts)}")

    def on_save(self, args, state, control, **kwargs):
        print(f"  [SAVE] Checkpoint salvato allo step {state.global_step}.")

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass   # Trainer's tqdm already shows per-step info


class CheckpointCleanupCallback(TrainerCallback):
    """Delete stale checkpoints immediately after each save.

    HF Trainer's save_total_limit only prunes checkpoints at the *next* save
    event, so on Kaggle (20 GB disk) you can temporarily have N+1 checkpoints
    on disk.  This callback removes every checkpoint folder except the most
    recently written one right after the Trainer finishes saving, keeping disk
    usage at roughly 1 × checkpoint size at all times.
    """

    def on_save(self, args, state, control, **kwargs):
        import glob, shutil
        output_dir = args.output_dir
        # Trainer names checkpoints as checkpoint-<step>
        ckpt_dirs = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[-1]),
        )
        # Keep only the last one (the one just saved)
        for old_ckpt in ckpt_dirs[:-1]:
            shutil.rmtree(old_ckpt, ignore_errors=True)
            print(f"  [CLEAN] Removed stale checkpoint: {os.path.basename(old_ckpt)}")

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_NAME   = os.environ.get("MODEL_PATH",   "MCG-NJU/videomae-base-finetuned-kinetics")
DATASET_PATH = os.environ.get("DATASET_PATH", "./distraction_dataset")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",   "./videomae_outputs")
HF_REPO_ID   = os.environ.get("HF_REPO_ID",  "")   # e.g. "username/videomae-distraction"

NUM_FRAMES = 16     # VideoMAE-Base native frame count
LIMIT_CAP  = 160    # max clips per class (None = no cap)
SEED       = 42

# Memory tuning via env vars
# Kaggle P100 (16 GB): TRAIN_BATCH_SIZE=2 GRAD_ACCUM_STEPS=8  → effective 16
# Kaggle T4   (16 GB): TRAIN_BATCH_SIZE=1 GRAD_ACCUM_STEPS=16 → effective 16
# CINECA A100 (40 GB): TRAIN_BATCH_SIZE=8 GRAD_ACCUM_STEPS=2  → effective 16
PER_DEVICE_TRAIN_BATCH = int(os.environ.get("TRAIN_BATCH_SIZE", "2"))
PER_DEVICE_EVAL_BATCH  = int(os.environ.get("EVAL_BATCH_SIZE",  "2"))
GRAD_ACCUM_STEPS       = int(os.environ.get("GRAD_ACCUM_STEPS", "8"))

CLASS_MAP = {
    "safe_driving":    0,
    "texting_right":   1,
    "phonecall_right": 2,
    "texting_left":    3,
    "phonecall_left":  4,
    "radio":           5,
    "drinking":        6,
    "reach_side":      7,
    "hair_and_makeup": 8,
    "change_gear":     9,
}
ID2LABEL    = {v: k for k, v in CLASS_MAP.items()}
NUM_CLASSES = len(CLASS_MAP)

# ── Dataset utilities ──────────────────────────────────────────────────────────

def build_entries(dataset_path, class_map, limit_cap=None, seed=42):
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
    """Loads video clips and returns pixel_values for VideoMAE.

    VideoMAE processor handles its own resizing (224×224) and normalisation,
    so no manual transforms are needed here.
    """
    def __init__(self, entries, processor, num_frames=16):
        self.entries    = entries
        self.processor  = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        video_path, label = self.entries[idx]
        frames = self._sample_frames(video_path)
        # VideoMAEImageProcessor expects a list of PIL Images
        inputs = self.processor(images=frames, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels":       torch.tensor(label, dtype=torch.long),
        }

    def _sample_frames(self, video_path):
        cap   = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self.num_frames

        indices = np.linspace(0, max(total - 1, 0), self.num_frames, dtype=int)
        frames  = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()

        if not frames:
            frames = [Image.new("RGB", (224, 224))] * self.num_frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1])
        return frames[:self.num_frames]

# ── Metrics ────────────────────────────────────────────────────────────────────

_accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)

    acc = _accuracy_metric.compute(
        predictions=predictions, references=labels
    )["accuracy"]

    top3_indices = np.argsort(logits, axis=-1)[:, -3:]
    top3_acc = float(np.mean([labels[i] in top3_indices[i] for i in range(len(labels))]))

    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)

    return {
        "accuracy":      round(acc,      4),
        "top3_accuracy": round(top3_acc, 4),
        "f1_weighted":   round(f1,       4),
    }

# ── Balanced Trainer ───────────────────────────────────────────────────────────

def compute_class_weights(train_entries, num_classes, device):
    """Compute inverse-frequency class weights and move them to device.

    These weights are passed to F.cross_entropy so that rarer classes
    (including safe_driving when it is under-represented) contribute
    more to the loss, preventing the model from ignoring them.
    """
    label_counts = Counter(label for _, label in train_entries)
    # Use the total number of training samples divided by (num_classes * count)
    # — same formula as sklearn's compute_class_weight('balanced')
    total = sum(label_counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float)
    for cls_id in range(num_classes):
        count = label_counts.get(cls_id, 1)   # avoid division by zero
        weights[cls_id] = total / (num_classes * count)
    print("Class weights for cross-entropy loss:")
    for cls_id, w in enumerate(weights):
        print(f"  [{cls_id:2d}] {ID2LABEL[cls_id]:25s}: {w:.4f}")
    return weights.to(device)


class BalancedTrainer(Trainer):
    """WeightedRandomSampler + class-weighted cross-entropy loss.

    The sampler ensures every class appears at roughly equal frequency
    within each batch.  The weighted loss provides an additional signal
    to the gradient so the model genuinely learns rare / hard classes
    (e.g. safe_driving) instead of collapsing them.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights   # Tensor on the correct device

    # ── Weighted loss ──────────────────────────────────────────────────────────
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss = F.cross_entropy(logits, labels, weight=self._class_weights)
        return (loss, outputs) if return_outputs else loss

    # ── Balanced sampler ───────────────────────────────────────────────────────
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

    print(f"Model      : {MODEL_NAME}")
    print(f"Dataset    : {DATASET_PATH}")
    print(f"Output     : {OUTPUT_DIR}")
    print(f"Frames     : {NUM_FRAMES}")
    print(f"Limit cap  : {LIMIT_CAP}")
    print(f"HF Repo    : {HF_REPO_ID or '(disabled — saving locally only)'}")
    print()

    # ── Load processor & model ────────────────────────────────────────────────
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    model = VideoMAEForVideoClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=ID2LABEL,
        label2id=CLASS_MAP,
        ignore_mismatched_sizes=True,   # replaces Kinetics classification head
    )

    # ── Build splits ──────────────────────────────────────────────────────────
    print("Scanning dataset...")
    entries = build_entries(DATASET_PATH, CLASS_MAP, limit_cap=LIMIT_CAP, seed=SEED)
    if not entries:
        raise RuntimeError(
            f"No video files found under {DATASET_PATH}. "
            "Did you run download_assets.py first?"
        )

    train_entries, val_entries, test_entries = stratified_split(
        entries, train_ratio=0.7, val_ratio=0.2, seed=SEED
    )
    print(f"\nSplit — Train: {len(train_entries)} | Val: {len(val_entries)} | Test: {len(test_entries)}")

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

        # Batch / accumulation
        # VideoMAE-Base uses less VRAM than TimeSformer-HR → higher batch possible
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        gradient_checkpointing=True,
        fp16=True,

        # Optimizer & scheduler
        learning_rate=5e-5,         # slightly higher than TimeSformer (MAE features are robust)
        num_train_epochs=15,        # more epochs — VideoMAE benefits from longer fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.05,

        # Logging & evaluation
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        # Keep only the single best checkpoint on disk — critical on Kaggle (20 GB limit)
        save_total_limit=1,

        dataloader_num_workers=2,
        disable_tqdm=False,

        # HF Hub push (optional)
        push_to_hub     = bool(HF_REPO_ID),
        hub_model_id    = HF_REPO_ID or None,
        hub_strategy    = "checkpoint",
        hub_private_repo= True,

        report_to="wandb",
        run_name=f"videomae-base-{NUM_FRAMES}frames",
        seed=SEED,
    )

    # ── Class weights (for weighted cross-entropy) ────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_weights = compute_class_weights(train_entries, NUM_CLASSES, device)

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            CustomLoggingCallback(),
            CheckpointCleanupCallback(),
        ],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    torch.cuda.empty_cache()

    print(f"\nStarting training...")
    print(f"  Batch/device={PER_DEVICE_TRAIN_BATCH}  GradAccum={GRAD_ACCUM_STEPS}  "
          f"EffectiveBatch={PER_DEVICE_TRAIN_BATCH * GRAD_ACCUM_STEPS}")
    trainer.train()

    # ── Test set evaluation ───────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_results = trainer.predict(test_dataset)
    print("Test metrics:")
    for k, v in test_results.metrics.items():
        print(f"  {k}: {v}")

    # Per-class report (precision / recall / f1 per ogni classe)
    preds  = np.argmax(test_results.predictions, axis=1)
    labels = test_results.label_ids
    print("\nPer-class classification report:")
    print(classification_report(
        labels, preds,
        target_names=[ID2LABEL[i] for i in range(NUM_CLASSES)],
        zero_division=0,
    ))

    # ── Save best model ───────────────────────────────────────────────────────
    best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(best_model_dir)
    processor.save_pretrained(best_model_dir)
    print(f"\nBest model saved to: {best_model_dir}")

    # ── Confusion matrix (saved inside best_model/) ───────────────────────────
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)],
                yticklabels=[ID2LABEL[i] for i in range(NUM_CLASSES)])
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix — VideoMAE Test Set")
    plt.tight_layout()
    cm_path = os.path.join(best_model_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # ── Training curves JSON (for plotting) ───────────────────────────────────
    # Extract one entry per epoch with train_loss, val_loss, train_acc, val_acc.
    # The HF Trainer writes training and eval metrics as *separate* entries in
    # log_history; we merge them by epoch number.
    epoch_data: dict[int, dict] = {}
    for entry in trainer.state.log_history:
        ep = int(entry.get("epoch", -1))
        if ep < 0:
            continue
        row = epoch_data.setdefault(ep, {"epoch": ep})
        if "loss" in entry:           row["train_loss"] = round(entry["loss"], 6)
        if "eval_loss" in entry:      row["val_loss"]   = round(entry["eval_loss"], 6)
        if "eval_accuracy" in entry:  row["val_acc"]    = round(entry["eval_accuracy"], 6)
        # train accuracy is not logged by default; include it when available
        if "train_accuracy" in entry: row["train_acc"]  = round(entry["train_accuracy"], 6)

    training_curves = sorted(epoch_data.values(), key=lambda r: r["epoch"])
    curves_path = os.path.join(best_model_dir, "training_curves.json")
    with open(curves_path, "w") as f:
        json.dump(training_curves, f, indent=4)
    print(f"Training curves saved to {curves_path}")

    # Full raw log (for debugging)
    raw_log_path = os.path.join(best_model_dir, "training_log_raw.json")
    with open(raw_log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"Raw training log saved to {raw_log_path}")

    if HF_REPO_ID:
        print(f"Checkpoints also pushed to HF Hub: https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
