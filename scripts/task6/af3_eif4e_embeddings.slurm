#!/bin/bash

#SBATCH --job-name=t6_af3_embed
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/task6_logs/t6_af3_embed_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/task6_logs/t6_af3_embed_%j.err

set -euo pipefail

echo "===== TASK 6 AF3 EMBEDDING EXPORT ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"

module load singularityce/4.1.0

CONTAINER=/projects/bentosprg6/linwx/containers/alphafold3_current_sandbox
WORK=/projects/bentosprg6/linwx/research_project
MODELS=/projects/bentosprg6/linwx/models/alphafold3
RUNTIME=/projects/bentosprg6/linwx/task6_runtime

INPUT=/work/af3_inputs/utexas_eif4e_apt1_no_msa.json
OUTPUT=/task6/af3_output/eif4e_apt1_embeddings

mkdir -p "${RUNTIME}/af3_output/eif4e_apt1_embeddings"

echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "===== RUNNING AF3 WITH EMBEDDING EXPORT ====="

singularity exec --userns --nv \
    -B ${WORK}:/work \
    -B ${MODELS}:/models:ro \
    -B ${RUNTIME}:/task6 \
    ${CONTAINER} \
    bash -lc 'cd /app/alphafold && python3 run_alphafold.py \
        --json_path=/work/af3_inputs/utexas_eif4e_apt1_no_msa.json \
        --model_dir=/models \
        --output_dir=/task6/af3_output/eif4e_apt1_embeddings \
        --run_data_pipeline=false \
        --run_inference=true \
        --save_embeddings=true'

echo
echo "===== EMBEDDING FILES ====="
find "${RUNTIME}/af3_output/eif4e_apt1_embeddings" \
    -type f \
    -name "*embeddings.npz" \
    -ls

echo
echo "===== FINISHED ====="
echo "End: $(date)"
