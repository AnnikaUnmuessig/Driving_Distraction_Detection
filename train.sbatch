#!/bin/bash
# =============================================================================
# train.sbatch — SLURM job script for CINECA Leonardo cluster
#
# BEFORE SUBMITTING:
#   1. Run download_assets.py on the LOGIN NODE (has internet access):
#        python download_assets.py --output_dir $WORK/distraction_detection/hf_cache
#
#   2. Fill in YOUR_ACCOUNT below with your CINECA project account code.
#
#   3. Update BASE_DIR if needed.
#
#   4. Submit:
#        sbatch train.sbatch
# =============================================================================

#SBATCH --job-name=timesformer-hr
#SBATCH --account=YOUR_ACCOUNT           # <-- Replace with your CINECA account (e.g. IscrC_MyProject)
#SBATCH --partition=boost_usr_prod       # A100 80GB GPUs on Leonardo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # 1 x A100 (add more if using DDP)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00                  # Adjust to your expected runtime
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Paths — edit BASE_DIR to match your $WORK structure ───────────────────────
BASE_DIR=$WORK/distraction_detection

MODEL_PATH=$BASE_DIR/hf_cache/timesformer-hr
DATASET_PATH=$BASE_DIR/hf_cache/distraction_dataset
OUTPUT_DIR=$BASE_DIR/outputs/timesformer-hr-16

# ── Environment setup ─────────────────────────────────────────────────────────
# Load the CINECA AI module stack (provides Python 3.10, PyTorch, CUDA 12.x)
module purge
module load profile/deeplrn
module load cineca-ai/4.3.0

# Activate your virtual environment if you installed extra packages (e.g. evaluate, scikit-learn)
# If you used pip install --user you can skip this line.
# source $BASE_DIR/venv/bin/activate

# ── Offline mode: compute nodes have NO internet access ───────────────────────
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# Point HuggingFace to the pre-downloaded cache
export HF_HOME=$BASE_DIR/hf_cache
export TRANSFORMERS_CACHE=$MODEL_PATH     # where timesformer-hr lives

# ── Pass paths to Finetuning.py via env vars ──────────────────────────────────
export MODEL_PATH=$MODEL_PATH
export DATASET_PATH=$DATASET_PATH
export OUTPUT_DIR=$OUTPUT_DIR

# ── W&B (offline — syncs when you run `wandb sync` from login node) ───────────
export WANDB_MODE=offline
export WANDB_PROJECT=driving_distraction
export WANDB_DIR=$BASE_DIR/wandb

# ── Performance tuning ────────────────────────────────────────────────────────
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# ── Create output directories ─────────────────────────────────────────────────
mkdir -p logs "$OUTPUT_DIR" "$WANDB_DIR"

# ── Diagnostics ───────────────────────────────────────────────────────────────
echo "=============================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "Python:   $(python --version)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Paths:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  DATASET_PATH=$DATASET_PATH"
echo "  OUTPUT_DIR=$OUTPUT_DIR"
echo "=============================="

# ── Run training ──────────────────────────────────────────────────────────────
cd $BASE_DIR  # assumes Finetuning.py is in $WORK/distraction_detection/
              # or adjust to: cd /path/to/Driving_Distraction_Detection

python Finetuning.py

echo "=============================="
echo "Finished: $(date)"
echo "=============================="
