#!/bin/bash
#SBATCH --job-name=em_tuning_report
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/em_tuning_report_%j.out
#SBATCH --error=logs/em_tuning_report_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
GRID_ROOT="${GRID_ROOT:-${EXPERIMENT_DIR}/em_grid_search}"
GRID_PROFILE="${GRID_PROFILE:-coarse}"
REPORT_OUT="${REPORT_OUT:-${GRID_ROOT}/em_tuning_report_${GRID_PROFILE}_$(date +%Y%m%d_%H%M%S).txt}"
TAIL_LINES="${TAIL_LINES:-80}"

# Comma-separated list of log files. You can override at submit time.
EM_LOG_FILES="${EM_LOG_FILES:-logs/em_grid_14633260.out,logs/em_merge_14633446.out,logs/em_remove_14633454.out,logs/em_add_14633456.out}"

cd "${REPO_DIR}"
mkdir -p logs

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

EXTRA_ARGS=()
IFS=',' read -r -a LOG_ARRAY <<< "${EM_LOG_FILES}"
for lf in "${LOG_ARRAY[@]}"; do
  if [ -n "${lf}" ]; then
    EXTRA_ARGS+=(--log-file "${lf}")
  fi
done

python src/scripts/em_grid_tuning_report.py \
  --grid_root "${GRID_ROOT}" \
  --profile "${GRID_PROFILE}" \
  --tail_lines "${TAIL_LINES}" \
  --output_file "${REPORT_OUT}" \
  "${EXTRA_ARGS[@]}"

echo "Report saved to: ${REPORT_OUT}"
