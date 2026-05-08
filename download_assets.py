"""
download_assets.py
==================
Run this script ONCE on the CINECA login node (which has internet access)
BEFORE submitting the training job via sbatch.
Also usable on Google Colab / Kaggle to download a limited subset of the dataset.

It downloads:
  1. The HuggingFace model  : facebook/timesformer-hr-finetuned-k400
  2. The HuggingFace dataset : endoard/distraction_detection_dataset
     (optionally limited to --videos_per_class N videos per class folder)

Usage:
    # Full dataset (~13 GB):
    python download_assets.py --output_dir /path/to/storage

    # Partial dataset — 50 videos per class:
    python download_assets.py --output_dir /path/to/storage --videos_per_class 50

Example on CINECA Leonardo ($WORK is persistent storage):
    python download_assets.py --output_dir $WORK/distraction_detection/hf_cache
"""

import argparse
import os
import random
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download, hf_hub_download, HfApi
import huggingface_hub.utils as hf_utils

MODEL_REPO   = "facebook/timesformer-hr-finetuned-k400"
DATASET_REPO = "endoard/distraction_detection_dataset"

# Classes expected in the dataset (folder names)
CLASS_NAMES = [
    "safe_driving",
    "texting_right",
    "phonecall_right",
    "texting_left",
    "phonecall_left",
    "radio",
    "drinking",
    "reach_side",
    "hair_and_makeup",
    "talking_to_passenger",
    "change_gear",
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


def download_model(output_dir):
    model_dir = os.path.join(output_dir, "timesformer-hr")
    os.makedirs(model_dir, exist_ok=True)
    print(f"\n[1/2] Downloading model '{MODEL_REPO}' -> {model_dir}")
    snapshot_download(
        repo_id=MODEL_REPO,
        repo_type="model",
        local_dir=model_dir,
        # Skip TF/JAX weights to save space
        ignore_patterns=["*.msgpack", "*.h5", "flax_model*", "tf_model*"],
    )
    print(f"      Model saved to: {model_dir}")
    return model_dir


def download_dataset_full(output_dir):
    """Download the entire dataset (~13 GB)."""
    dataset_dir = os.path.join(output_dir, "distraction_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"\n[2/2] Downloading FULL dataset '{DATASET_REPO}' -> {dataset_dir}")
    print("      This is ~13 GB, may take a while...")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print(f"      Dataset saved to: {dataset_dir}")
    return dataset_dir


def download_dataset_partial(output_dir, videos_per_class, seed=42):
    """
    Download only `videos_per_class` videos per class folder.

    Strategy:
      1. List all files in the HF dataset repo.
      2. Group them by top-level folder (= class name).
      3. Shuffle and keep only `videos_per_class` per class.
      4. Download each file individually via hf_hub_download,
         using a single tqdm bar (HF's per-file bars are suppressed).
    """
    dataset_dir = os.path.join(output_dir, "distraction_dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    print(f"\n[2/2] Listing files in dataset repo '{DATASET_REPO}'...")
    api = HfApi()
    all_files = list(api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset"))

    # Group by class folder
    rng = random.Random(seed)
    by_class: dict[str, list[str]] = {cls: [] for cls in CLASS_NAMES}

    for filepath in all_files:
        parts = filepath.replace("\\", "/").split("/")
        if len(parts) < 2:
            continue
        top_folder = parts[0]
        ext = os.path.splitext(parts[-1])[1].lower()
        if top_folder in by_class and ext in VIDEO_EXTENSIONS:
            by_class[top_folder].append(filepath)

    # Build flat download list and print per-class summary
    plan: dict[str, list[str]] = {}
    flat_list: list[tuple[str, str]] = []   # (class_name, repo_path)
    for cls, files in by_class.items():
        rng.shuffle(files)
        selected = files[:videos_per_class]
        plan[cls] = selected
        flat_list.extend((cls, p) for p in selected)
        print(f"  {cls:25s}: {len(selected):4d} / {len(files)} videos selected")

    total = len(flat_list)
    print(f"\n  Total: {total} videos  |  Destination: {dataset_dir}")

    # ── Single progress bar for the entire download ────────────────────────────
    downloaded = 0
    errors = 0
    with tqdm(total=total, unit="video", desc="Downloading", dynamic_ncols=True) as pbar:
        for cls, repo_path in flat_list:
            filename = os.path.basename(repo_path)
            dest_path = os.path.join(dataset_dir, cls, filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Update description before each download so class is visible
            pbar.set_description(f"Downloading [{cls}]")

            if os.path.exists(dest_path):
                downloaded += 1
                pbar.update(1)
                continue

            try:
                hf_hub_download(
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    filename=repo_path,
                    local_dir=dataset_dir,
                )
                downloaded += 1
            except Exception as e:
                errors += 1
                tqdm.write(f"[WARN] Failed: {repo_path} — {e}")

            pbar.update(1)

    status = f"{downloaded}/{total} downloaded"
    if errors:
        status += f", {errors} errors"
    print(f"\n  Done. {status}. Saved to {dataset_dir}")
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model and dataset for offline training."
    )
    parser.add_argument(
        "--output_dir",
        default="./hf_cache",
        help="Root directory where model and dataset will be saved. "
             "On CINECA, use $WORK or $SCRATCH. (default: ./hf_cache)",
    )
    parser.add_argument(
        "--videos_per_class",
        type=int,
        default=None,
        metavar="N",
        help="Download only N videos per class folder instead of the full dataset. "
             "Useful on Colab/Kaggle to avoid downloading ~13 GB. "
             "(default: None = download everything)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for file selection when --videos_per_class is set. (default: 42)",
    )
    args = parser.parse_args()

    # Disable ALL huggingface_hub internal progress bars (per-chunk tqdm).
    # Must be called here, after import, before any download — env vars are too late.
    hf_utils.disable_progress_bars()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Storage root: {output_dir}")

    model_dir = download_model(output_dir)

    if args.videos_per_class is not None:
        dataset_dir = download_dataset_partial(
            output_dir,
            videos_per_class=args.videos_per_class,
            seed=args.seed,
        )
    else:
        dataset_dir = download_dataset_full(output_dir)

    # Print the env vars to copy into train.sbatch / notebook
    print("\n" + "=" * 60)
    print("Download complete. Set these paths in your training script:")
    print("=" * 60)
    print(f'  export MODEL_PATH="{model_dir}"')
    print(f'  export DATASET_PATH="{dataset_dir}"')
    print(f'  export OUTPUT_DIR="{os.path.join(output_dir, "outputs/timesformer-hr-16")}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
