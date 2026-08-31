#!/bin/bash

#SBATCH --job-name=af3_data
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/af3_logs/af3_data_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/af3_logs/af3_data_%j.err

set -e

echo "===== AF3 DATA PIPELINE ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

AF3_DIR=/projects/bentosprg6/linwx/repos/alphafold3
INPUT=/projects/bentosprg6/linwx/research_project/af3_inputs/tetr_homodimer.json
OUTPUT=/projects/bentosprg6/linwx/research_project/af3_output

# WAITING FOR LAB/WEI TO PROVIDE THIS:
DB_DIR=/PATH/TO/APPROVED/AF3_DATABASES

cd "$AF3_DIR"

# Container/environment command will be inserted here after Wei
# confirms the supported Andromeda setup.

python run_alphafold.py \
    --json_path="$INPUT" \
    --db_dir="$DB_DIR" \
    --output_dir="$OUTPUT" \
    --run_data_pipeline=true \
    --run_inference=false \
    --force_output_dir=true

echo "Finished: $(date)"
