#!/bin/bash

#SBATCH --job-name=af3_adr7
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/af3_logs/af3_adr7_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/af3_logs/af3_adr7_%j.err

set -euo pipefail

echo "===== AF3 UTEXAS ADR7 + CCL2 ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"

module load singularityce/4.1.0

CONTAINER=/projects/bentosprg6/linwx/containers/alphafold3_current_sandbox
WORK=/projects/bentosprg6/linwx/research_project
MODELS=/projects/bentosprg6/linwx/models/alphafold3

mkdir -p "${WORK}/af3_output"
mkdir -p "${WORK}/af3_logs"

echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "===== Running AlphaFold 3 ====="

singularity exec --userns --nv \
    -B ${WORK}:/work \
    -B ${MODELS}:/models:ro \
    ${CONTAINER} \
    bash -lc 'cd /app/alphafold && python3 run_alphafold.py \
        --json_path=/work/af3_inputs/utexas_adr7_ccl2_no_msa.json \
        --model_dir=/models \
        --output_dir=/work/af3_output/utexas_adr7_ccl2_no_msa \
        --run_data_pipeline=false \
        --run_inference=true'

echo "===== Finished ====="
echo "End time: $(date)"
