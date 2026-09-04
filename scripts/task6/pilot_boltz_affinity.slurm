#!/bin/bash

#SBATCH --job-name=t6_boltzpilot
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/task6_logs/t6_boltzpilot_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/task6_logs/t6_boltzpilot_%j.err

set -euo pipefail

module load miniconda/3

source activate /projects/bentosprg6/linwx/envs/boltz

export PYTHONPATH=/projects/bentosprg6/linwx/repos/boltz/src:${PYTHONPATH:-}

cd /projects/bentosprg6/linwx/research_project

TASK6_INDEX="${TASK6_INDEX:-${SLURM_ARRAY_TASK_ID:-}}"

if [ -z "${TASK6_INDEX}" ]; then
    echo "ERROR: Neither TASK6_INDEX nor SLURM_ARRAY_TASK_ID was supplied."
    exit 1
fi

echo "===== TASK 6 GENERIC AF3 -> BOLTZ AFFINITY ====="
echo "Job: $SLURM_JOB_ID"
echo "Index: $TASK6_INDEX"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

nvidia-smi --query-gpu=name,memory.total --format=csv

echo
echo "===== RUNNING AFFINITY ====="

python scripts/task6/run_pilot_af3_boltz_affinity.py \
    --task6-index "$TASK6_INDEX"

echo
echo "===== FINISHED ====="
echo "End: $(date)"
