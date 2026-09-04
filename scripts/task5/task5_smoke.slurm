#!/bin/bash
#SBATCH --job-name=t5_smoke
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/logs/task5/task5_smoke_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/logs/task5/task5_smoke_%j.err

set -euo pipefail

RUN_ID="${1:?Usage: sbatch task5_smoke.slurm task5_XXXX}"

PROJECT=/projects/bentosprg6/linwx/research_project
BOLTZ_REPO=/projects/bentosprg6/linwx/repos/boltz

INPUT="$PROJECT/data/task5/boltz_inputs/${RUN_ID}.yaml"
OUTDIR="$PROJECT/results/task5/smoke/${RUN_ID}_${SLURM_JOB_ID}"

echo "========== TASK 5 SMOKE TEST =========="
echo "Run ID: $RUN_ID"
echo "Input:  $INPUT"
echo "Output: $OUTDIR"
echo "Host:   $(hostname)"
echo "Start:  $(date)"

test -f "$INPUT"

module load miniconda/3
conda activate /projects/bentosprg6/linwx/envs/boltz

export BOLTZ_CACHE=/projects/bentosprg6/linwx/cache/boltz
export TMPDIR=/scratch/linwx/tmp
export PYTHONPATH="$BOLTZ_REPO/src:${PYTHONPATH:-}"

mkdir -p "$BOLTZ_CACHE" "$TMPDIR" "$OUTDIR"

echo
echo "========== VERIFY TASK 4 PATCH =========="

python - <<'PY'
from pathlib import Path
import boltz
from boltz.data.parse import schema

print("boltz:", boltz.__file__)
print("schema:", schema.__file__)

text = Path(schema.__file__).read_text()

assert "TASK 4: permit a DNA/RNA polymer" in text, \
    "ERROR: Task-4 aptamer affinity patch is not active."

print("TASK 4 PATCH CONFIRMED")
PY

echo
echo "========== INPUT =========="
cat "$INPUT"

echo
echo "========== RUN BOLTZ =========="

boltz predict "$INPUT" \
    --out_dir "$OUTDIR" \
    --devices 1 \
    --override

echo
echo "========== AFFINITY OUTPUTS =========="

find "$OUTDIR" -type f \( \
    -iname '*affinity*.json' -o \
    -iname '*affinity*.csv' \
\) -print

echo
echo "========== FINISHED =========="
echo "End: $(date)"
