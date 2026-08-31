#!/bin/bash

#SBATCH --job-name=af3_infer
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/af3_logs/af3_infer_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/af3_logs/af3_infer_%j.err

set -e

echo "===== AF3 INFERENCE ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

AF3_DIR=/projects/bentosprg6/linwx/repos/alphafold3
OUTPUT=/projects/bentosprg6/linwx/research_project/af3_output

# WAITING FOR APPROVED MODEL LOCATION:
MODEL_DIR=/PATH/TO/APPROVED/AF3_MODELS

# The CPU pipeline will generate this augmented JSON.
INPUT=/projects/bentosprg6/linwx/research_project/af3_output/tetr_homodimer_data.json

cd "$AF3_DIR"

# Container/environment command will be inserted here after Wei
# confirms the supported Andromeda setup.

python run_alphafold.py \
    --json_path="$INPUT" \
    --model_dir="$MODEL_DIR" \
    --output_dir="$OUTPUT" \
    --run_data_pipeline=false \
    --run_inference=true \
    --force_output_dir=true

echo "Finished: $(date)"
