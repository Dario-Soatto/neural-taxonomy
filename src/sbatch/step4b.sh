#!/bin/bash
#SBATCH --job-name=attach_text_wiki
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/attach_text_%j.out
#SBATCH --error=logs/attach_text_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
CLUSTERED_FILE="${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters.csv"
RAW_FILE="${REPO_DIR}/processed_custom_input.csv"
OUTPUT_FILE="${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"

cd "${REPO_DIR}"
mkdir -p logs

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

python src/step_4b__attach_sentence_text.py \
  --clustered_file "${CLUSTERED_FILE}" \
  --raw_sentences_file "${RAW_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --index_col index \
  --raw_doc_col doc_index \
  --raw_sent_col sent_index \
  --raw_text_col sentence_text \
  --out_text_col sentence_text
