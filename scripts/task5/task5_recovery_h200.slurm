#!/bin/bash
#SBATCH --job-name=t5r_h200
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/logs/task5/recovery_h200_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/logs/task5/recovery_h200_%j.err

set -euo pipefail

PROJECT=/projects/bentosprg6/linwx/research_project
BOLTZ_REPO=/projects/bentosprg6/linwx/repos/boltz

RUN_ID=task5r_0075
INPUT="$PROJECT/data/task5/recovery_inputs/${RUN_ID}.yaml"
OUTDIR="$PROJECT/results/task5/recovery_large/${RUN_ID}"

echo "========== TASK 5 RECOVERY H200 =========="
echo "Run ID : $RUN_ID"
echo "Input  : $INPUT"
echo "Output : $OUTDIR"
echo "Host   : $(hostname)"
echo "Start  : $(date)"

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

nvidia-smi --query-gpu=name,memory.total,memory.free \
  --format=csv,noheader || true

boltz predict "$INPUT" \
    --out_dir "$OUTDIR" \
    --devices 1 \
    --override

echo
echo "========== AFFINITY FILE =========="
find "$OUTDIR" \
    -type f \
    -name "affinity_${RUN_ID}.json" \
    -print

echo
echo "========== FINISHED =========="
echo "End: $(date)"
