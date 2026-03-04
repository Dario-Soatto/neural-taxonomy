#!/bin/bash
#SBATCH --job-name=em_grid_snapshot
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/em_grid_snapshot_%j.out
#SBATCH --error=logs/em_grid_snapshot_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
GRID_ROOT="${GRID_ROOT:-experiments/wiki_biographies_10000/em_grid_search}"
GRID_OPERATION="${GRID_OPERATION:-split}"
GRID_PROFILE="${GRID_PROFILE:-coarse}"
LOGS_GLOB="${LOGS_GLOB:-logs/em_*.out}"
OUT_DIR="${OUT_DIR:-logs}"
TOP_K="${TOP_K:-15}"

cd "${REPO_DIR}"
mkdir -p logs

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

python src/scripts/em_grid_operation_snapshot.py \
  --grid_root "${GRID_ROOT}" \
  --operation "${GRID_OPERATION}" \
  --profile "${GRID_PROFILE}" \
  --logs_glob "${LOGS_GLOB}" \
  --output_dir "${OUT_DIR}" \
  --top_k "${TOP_K}"
