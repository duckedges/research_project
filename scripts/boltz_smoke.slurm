#!/bin/bash

#SBATCH --job-name=boltz_smoke
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/logs/boltz_smoke_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/logs/boltz_smoke_%j.err

# Exit immediately if a command fails
set -euo pipefail

echo "===== Job Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

echo "===== GPU Information ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Load Python environment
module load miniconda/3
conda activate /projects/bentosprg6/linwx/envs/boltz

# Keep caches outside home directory
export BOLTZ_CACHE=/projects/bentosprg6/linwx/cache/boltz
export PIP_CACHE_DIR=/projects/bentosprg6/linwx/cache/pip
export TMPDIR=/scratch/linwx/tmp

mkdir -p "$BOLTZ_CACHE"
mkdir -p "$TMPDIR"
OUTDIR=/projects/bentosprg6/linwx/research_project/results/boltz_batch_${SLURM_JOB_ID}
mkdir -p "$OUTDIR"

echo "===== Python / CUDA ====="

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "===== Running Boltz ====="

cd /projects/bentosprg6/linwx/repos/boltz

boltz predict examples/prot_no_msa.yaml \
    --out_dir "$OUTDIR" \
    --devices 1

echo "===== Finished ====="
echo "End time: $(date)"
