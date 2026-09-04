#!/bin/bash
#SBATCH --job-name=t5_boltz
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/logs/task5/batch_%A_%a.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/logs/task5/batch_%A_%a.err

set -euo pipefail

PROJECT=/projects/bentosprg6/linwx/research_project
BOLTZ_REPO=/projects/bentosprg6/linwx/repos/boltz

RUN_ID=$(printf "task5_%04d" "$SLURM_ARRAY_TASK_ID")

INPUT="$PROJECT/data/task5/boltz_inputs/${RUN_ID}.yaml"
OUTDIR="$PROJECT/results/task5/batch/${RUN_ID}"

echo "========== TASK 5 BOLTZ =========="
echo "Array job:  $SLURM_ARRAY_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Run ID:     $RUN_ID"
echo "Input:      $INPUT"
echo "Output:     $OUTDIR"
echo "Host:       $(hostname)"
echo "Start:      $(date)"

test -f "$INPUT"

module load miniconda/3
conda activate /projects/bentosprg6/linwx/envs/boltz

export BOLTZ_CACHE=/projects/bentosprg6/linwx/cache/boltz
export TMPDIR=/scratch/linwx/tmp
export PYTHONPATH="$BOLTZ_REPO/src:${PYTHONPATH:-}"

mkdir -p "$BOLTZ_CACHE" "$TMPDIR" "$OUTDIR"

python - <<'PY'
from pathlib import Path
from boltz.data.parse import schema

text = Path(schema.__file__).read_text()

assert "TASK 4: permit a DNA/RNA polymer" in text, \
    "ERROR: Task-4 aptamer-affinity patch is not active."

print("TASK 4 PATCH CONFIRMED")
PY

boltz predict "$INPUT" \
    --out_dir "$OUTDIR" \
    --devices 1 \
    --override

echo
echo "========== AFFINITY FILE =========="

find "$OUTDIR" -type f -name 'affinity_*.json' -print

echo
echo "========== FINISHED =========="
echo "End: $(date)"
