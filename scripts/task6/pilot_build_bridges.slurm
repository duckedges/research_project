#!/bin/bash

#SBATCH --job-name=t6_bridge
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/task6_logs/t6_bridge_%A_%a.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/task6_logs/t6_bridge_%A_%a.err

set -euo pipefail

cd /projects/bentosprg6/linwx/research_project

module load miniconda/3
source activate /projects/bentosprg6/linwx/envs/boltz

export PYTHONPATH=/projects/bentosprg6/linwx/repos/boltz/src:${PYTHONPATH:-}

echo "===== TASK 6 AF3 -> BOLTZ BRIDGE ====="
echo "Job: $SLURM_JOB_ID"
echo "Index: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo

python scripts/task6/build_pilot_af3_bridge.py \
    --task6-index "$SLURM_ARRAY_TASK_ID"

echo
echo "===== SUCCESS ====="
