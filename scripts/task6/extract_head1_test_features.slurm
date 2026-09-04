#!/bin/bash

#SBATCH --job-name=t6_testfeat
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/task6_logs/t6_testfeat_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/task6_logs/t6_testfeat_%j.err

set -euo pipefail

module load miniconda/3
source activate /projects/bentosprg6/linwx/envs/boltz

export PYTHONPATH=/projects/bentosprg6/linwx/repos/boltz/src:${PYTHONPATH:-}

cd /projects/bentosprg6/linwx/research_project

echo "===== TASK 6 HELD-OUT TEST FEATURE EXTRACTION ====="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo

nvidia-smi --query-gpu=name,memory.total --format=csv

echo
python scripts/task6/extract_head1_test_features.py

echo
echo "===== FINISHED ====="
echo "End: $(date)"
