#!/bin/bash
# Run all CPU-only baselines for newsgroups and/or wiki.
#
# Usage:
#   sbatch src/sbatch/baselines_cpu.sh
#   sbatch --export=ALL,DATASET=wiki src/sbatch/baselines_cpu.sh
#   sbatch --export=ALL,METHODS="lda_gibbs embedding_kmeans" src/sbatch/baselines_cpu.sh

#SBATCH --job-name=baselines_cpu
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baselines_cpu_%j.out
#SBATCH --error=logs/baselines_cpu_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
DATASET="${DATASET:-newsgroups}"
N_RUNS="${N_RUNS:-3}"
METHODS="${METHODS:-lda_gibbs lda_vi embedding_kmeans bertopic ctm dvae scholar}"

cd "${REPO_DIR}"
mkdir -p logs

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "========================================"
echo "Baseline CPU benchmark"
echo "Started:     $(date)"
echo "Dataset:     ${DATASET}"
echo "Methods:     ${METHODS}"
echo "N_runs:      ${N_RUNS}"
echo "========================================"

if [[ "${DATASET}" == "newsgroups" || "${DATASET}" == "both" ]]; then
  EXP_DIR="experiments/newsgroups_sanity_tiny"
  if [[ ! -f "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" ]]; then
    echo "ERROR: Newsgroups data not found. Run newsgroups_sanity_prep_john.sh first."
    exit 1
  fi
  python src/run_baseline_benchmark.py \
    --dataset newsgroups \
    --experiment_dir "${EXP_DIR}" \
    --methods ${METHODS} \
    --n_clusters 8 \
    --text_column description \
    --n_runs "${N_RUNS}" \
    --filter_top_k_categories 8 \
    --output_dir "${EXP_DIR}/baseline_results"
fi

if [[ "${DATASET}" == "wiki" || "${DATASET}" == "both" ]]; then
  EXP_DIR="experiments/wiki_biographies_10000"
  if [[ ! -f "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" ]]; then
    echo "ERROR: Wiki data not found at ${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"
    exit 1
  fi
  python src/run_baseline_benchmark.py \
    --dataset wiki \
    --experiment_dir "${EXP_DIR}" \
    --methods ${METHODS} \
    --n_clusters 10 \
    --text_column sentence_text \
    --n_runs "${N_RUNS}" \
    --output_dir "${EXP_DIR}/baseline_results"
fi

echo "========================================"
echo "Finished:    $(date)"
echo "========================================"
