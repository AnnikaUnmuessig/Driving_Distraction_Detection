"""
download_assets.py
==================
Run this script ONCE on the CINECA login node (which has internet access)
BEFORE submitting the training job via sbatch.

It downloads:
  1. The HuggingFace model  : facebook/timesformer-hr-finetuned-k400
  2. The HuggingFace dataset : endoard/distraction_dataset

Usage:
    python download_assets.py --output_dir /path/to/storage

Example on CINECA Leonardo ($WORK is persistent storage):
    python download_assets.py --output_dir $WORK/distraction_detection/hf_cache
"""

import argparse
import os
from huggingface_hub import snapshot_download

MODEL_REPO   = "facebook/timesformer-hr-finetuned-k400"
DATASET_REPO = "endoard/distraction_dataset"


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


def download_dataset(output_dir):
    dataset_dir = os.path.join(output_dir, "distraction_dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    print(f"\n[2/2] Downloading dataset '{DATASET_REPO}' -> {dataset_dir}")
    print("      This is ~13 GB, may take a while...")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=dataset_dir,
    )
    print(f"      Dataset saved to: {dataset_dir}")
    return dataset_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download HuggingFace model and dataset for offline CINECA training."
    )
    parser.add_argument(
        "--output_dir",
        default="./hf_cache",
        help="Root directory where model and dataset will be saved. "
             "On CINECA, use $WORK or $SCRATCH. (default: ./hf_cache)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Storage root: {output_dir}")

    model_dir   = download_model(output_dir)
    dataset_dir = download_dataset(output_dir)

    # Print the env vars to copy into train.sbatch
    print("\n" + "=" * 60)
    print("Download complete. Set these paths in train.sbatch:")
    print("=" * 60)
    print(f'  export MODEL_PATH="{model_dir}"')
    print(f'  export DATASET_PATH="{dataset_dir}"')
    print(f'  export OUTPUT_DIR="{os.path.join(output_dir, "outputs/timesformer-hr-16")}"')
    print("=" * 60)


if __name__ == "__main__":
    main()
