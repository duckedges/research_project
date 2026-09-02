#!/bin/bash
#SBATCH --job-name=t4_adr7_aff
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/logs/task4_adr7_aff_%j.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/logs/task4_adr7_aff_%j.err

set -euo pipefail

PROJECT=/projects/bentosprg6/linwx/research_project
BOLTZ_REPO=/projects/bentosprg6/linwx/repos/boltz
INPUT=$PROJECT/data/task4/boltz_inputs/adr7_ccl2_affinity.yaml
OUTDIR=$PROJECT/results/task4/adr7_ccl2_${SLURM_JOB_ID}

echo "========================================"
echo " TASK 4: APTAMER AFFINITY TEST"
echo "========================================"
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $(hostname)"
echo "Start time: $(date)"
echo "Input:      $INPUT"
echo "Output:     $OUTDIR"

echo
echo "========== GPU =========="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "========== ENVIRONMENT =========="
module load miniconda/3
conda activate /projects/bentosprg6/linwx/envs/boltz

export BOLTZ_CACHE=/projects/bentosprg6/linwx/cache/boltz
export TMPDIR=/scratch/linwx/tmp
mkdir -p "$BOLTZ_CACHE" "$TMPDIR" "$OUTDIR"

# CRITICAL:
# Force Python/Boltz to use our Task-4 patched source tree.
export PYTHONPATH="$BOLTZ_REPO/src:${PYTHONPATH:-}"

echo "Python: $(which python)"
echo "Boltz executable: $(which boltz)"

echo
echo "========== VERIFY PATCHED BOLTZ SOURCE =========="
python - <<'PY'
import boltz
import boltz.data.parse.schema as schema
from pathlib import Path

print("boltz package:")
print(boltz.__file__)

print("schema.py:")
print(schema.__file__)

text = Path(schema.__file__).read_text()

assert 'not in {"ligand", "rna", "dna"}' in text, \
    "ERROR: Task-4 patched parser is NOT being imported."

assert "TASK 4: permit a DNA/RNA polymer" in text, \
    "ERROR: Task-4 polymer-affinity patch is NOT being imported."

print("TASK 4 PATCH CONFIRMED")
PY

echo
echo "========== BOLTZ GIT VERSION =========="
cd "$BOLTZ_REPO"
git status --short --branch
git log -1 --oneline

echo
echo "========== INPUT =========="
cat "$INPUT"

echo
echo "========== START BOLTZ =========="
boltz predict "$INPUT" \
    --out_dir "$OUTDIR" \
    --devices 1 \
    --override

echo
echo "========== OUTPUT FILES =========="
find "$OUTDIR" -maxdepth 5 -type f | sort

echo
echo "========== AFFINITY OUTPUTS =========="
find "$OUTDIR" -type f \( \
    -iname "*affinity*.json" -o \
    -iname "*affinity*.csv" \
\) -print

echo
echo "========================================"
echo "TASK 4 RUN FINISHED"
echo "End time: $(date)"
echo "========================================"
