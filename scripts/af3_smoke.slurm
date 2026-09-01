#!/bin/bash
#SBATCH --job-name=af3_smoke
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/af3_smoke_%j.out
#SBATCH --error=logs/af3_smoke_%j.err

set -euo pipefail

echo "===== Job Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

module load singularityce/4.1.0

CONTAINER=/projects/bentosprg6/linwx/containers/alphafold3_current_sandbox
WORK=/projects/bentosprg6/linwx/research_project
MODELS=/projects/bentosprg6/linwx/models/alphafold3

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Running AlphaFold 3 ====="

singularity exec --userns --nv \
  -B ${WORK}:/work \
  -B ${MODELS}:/models:ro \
  ${CONTAINER} \
  bash -lc 'cd /app/alphafold && python3 run_alphafold.py \
    --json_path=/work/af3_inputs/tetr_homodimer_no_msa.json \
    --model_dir=/models \
    --output_dir=/work/af3_output/tetr_homodimer_no_msa_batch \
    --run_data_pipeline=false \
    --run_inference=true'

echo "===== Finished ====="
echo "End time: $(date)"
