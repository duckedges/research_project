#!/bin/bash

PROJECT=/projects/bentosprg6/linwx/research_project
STANDARD_JOB=2962367
LARGE_JOB=2962368
TOTAL=342

echo "========== TASK 5 PROGRESS =========="
echo "Time: $(date)"
echo

DONE=$(find "$PROJECT/results/task5/batch" \
  -type f -name 'affinity_task5_*.json' 2>/dev/null | wc -l)

echo "Affinity JSON completed: $DONE / $TOTAL"
echo

echo "========== STANDARD ARRAY =========="
squeue -r -j "$STANDARD_JOB" -h -o "%T" 2>/dev/null \
  | sort | uniq -c || true

echo
echo "========== LARGE ARRAY =========="
squeue -r -j "$LARGE_JOB" -h -o "%T" 2>/dev/null \
  | sort | uniq -c || true

echo
echo "========== ACTIVE TASKS =========="
squeue -r -j "${STANDARD_JOB},${LARGE_JOB}" \
  -o "%.18i %.10j %.2t %.10M %.18R" 2>/dev/null \
  | head -n 25

echo
echo "========== COMPLETED OUTPUT EXAMPLES =========="
find "$PROJECT/results/task5/batch" \
  -type f -name 'affinity_task5_*.json' 2>/dev/null \
  | sort | tail -n 10
