#!/bin/bash

#SBATCH --job-name=t6_af3pilot
#SBATCH --account=bentosprg6
#SBATCH --partition=short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/projects/bentosprg6/linwx/research_project/task6_logs/t6_af3pilot_%A_%a.out
#SBATCH --error=/projects/bentosprg6/linwx/research_project/task6_logs/t6_af3pilot_%A_%a.err

set -euo pipefail

module load singularityce/4.1.0

PROJECT=/projects/bentosprg6/linwx/research_project
INPUT_DIR=${PROJECT}/data/task6/af3_inputs

CONTAINER=/projects/bentosprg6/linwx/containers/alphafold3_current_sandbox
MODELS=/projects/bentosprg6/linwx/models/alphafold3
RUNTIME=/projects/bentosprg6/linwx/task6_runtime

IDX=${SLURM_ARRAY_TASK_ID}

PREFIX=$(printf "task6_%04d_" "${IDX}")

JSON=$(find "${INPUT_DIR}" \
    -maxdepth 1 \
    -type f \
    -name "${PREFIX}*.json" \
    -print \
    -quit)

if [ -z "${JSON}" ]; then
    echo "ERROR: no JSON found for array index ${IDX}"
    exit 1
fi

JSON_BASENAME=$(basename "${JSON}")
INPUT_NAME=${JSON_BASENAME%.json}

OUTPUT_ROOT=${RUNTIME}/pilot_af3
TASK_OUT=${OUTPUT_ROOT}/${INPUT_NAME}

# Remove only this individual case's previous runtime output.
rm -rf "${TASK_OUT}"
mkdir -p "${TASK_OUT}"

echo "===== TASK 6 AF3 PILOT EMBEDDING EXPORT ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array ID: ${IDX}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo
echo "Input: ${JSON}"
echo "Input name: ${INPUT_NAME}"
echo "Output: ${TASK_OUT}"

echo
echo "===== GPU ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo
echo "===== RUNNING AF3 ====="

singularity exec --userns --nv \
    -B "${INPUT_DIR}:/inputs:ro" \
    -B "${MODELS}:/models:ro" \
    -B "${RUNTIME}:/task6" \
    "${CONTAINER}" \
    bash -lc "cd /app/alphafold && python3 run_alphafold.py \
        --json_path=/inputs/${JSON_BASENAME} \
        --model_dir=/models \
        --output_dir=/task6/pilot_af3/${INPUT_NAME} \
        --run_data_pipeline=false \
        --run_inference=true \
        --save_embeddings=true"

echo
echo "===== OUTPUT QC ====="

EMBED=$(find "${TASK_OUT}" \
    -type f \
    -name "*embeddings.npz" \
    -print \
    -quit)

MODEL=$(find "${TASK_OUT}" \
    -type f \
    -name "*_model.cif" \
    -print \
    -quit)

if [ -z "${EMBED}" ]; then
    echo "ERROR: embeddings.npz not found."
    exit 2
fi

if [ -z "${MODEL}" ]; then
    echo "ERROR: model CIF not found."
    exit 3
fi

echo "Embedding: ${EMBED}"
ls -lh "${EMBED}"

echo
echo "Model:"
echo "${MODEL}"

echo "SUCCESS" > "${TASK_OUT}/TASK6_AF3_SUCCESS.txt"

echo
echo "===== SUCCESS ====="
echo "AF3 structure and embeddings generated for ${INPUT_NAME}"
echo "End: $(date)"
