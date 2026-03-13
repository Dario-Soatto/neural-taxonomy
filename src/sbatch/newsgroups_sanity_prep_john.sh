#!/bin/bash
#SBATCH --job-name=ng_prep_tiny
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/newsgroups_prep_%j.out
#SBATCH --error=logs/newsgroups_prep_%j.err

set -euo pipefail

# Allow overrides at submit time:
#   sbatch --export=ALL,N_DOCS=300,N_CENTROIDS=80,N_CLUSTERS=8 src/sbatch/newsgroups_sanity_prep_john.sh
REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
EXP_NAME="${EXP_NAME:-newsgroups_sanity_tiny}"
EXP_DIR="${EXP_DIR:-${REPO_DIR}/experiments/${EXP_NAME}}"

N_DOCS="${N_DOCS:-240}"
N_CENTROIDS="${N_CENTROIDS:-60}"
N_CLUSTERS="${N_CLUSTERS:-8}"
TOP_N_CATEGORIES="${TOP_N_CATEGORIES:-${N_CLUSTERS}}"
SEED="${SEED:-13}"
MAX_WORDS="${MAX_WORDS:-1200}"
SPLIT_SOURCE_LABEL="${SPLIT_SOURCE_LABEL:-}"
MERGE_KEEP_LABEL="${MERGE_KEEP_LABEL:-}"
MERGE_DROP_LABEL="${MERGE_DROP_LABEL:-}"
REMOVE_SOURCE_LABEL="${REMOVE_SOURCE_LABEL:-}"
ADD_SOURCE_LABEL="${ADD_SOURCE_LABEL:-}"
ADD_TARGET_LABELS="${ADD_TARGET_LABELS:-}"

cd "${REPO_DIR}"
mkdir -p logs "${EXP_DIR}/models"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "=== Newsgroups prep job ==="
echo "repo:       ${REPO_DIR}"
echo "exp_dir:    ${EXP_DIR}"
echo "n_docs:     ${N_DOCS}"
echo "n_centroid: ${N_CENTROIDS}"
echo "n_clusters: ${N_CLUSTERS}"
echo "top_n_cats: ${TOP_N_CATEGORIES}"
echo "seed:       ${SEED}"
echo "max_words:  ${MAX_WORDS}"
echo "==========================="

python src/scripts/prepare_newsgroups_data.py \
  --output_csv "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" \
  --n_docs "${N_DOCS}" \
  --n_centroids "${N_CENTROIDS}" \
  --max_words "${MAX_WORDS}" \
  --subset all \
  --seed "${SEED}"

HIERARCHY_ARGS=(
  --data_csv "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"
  --output_root "${EXP_DIR}"
  --top_n_categories "${TOP_N_CATEGORIES}"
  --seed "${SEED}"
)

if [[ -n "${SPLIT_SOURCE_LABEL}" ]]; then
  HIERARCHY_ARGS+=(--split_source_label "${SPLIT_SOURCE_LABEL}")
fi
if [[ -n "${MERGE_KEEP_LABEL}" ]]; then
  HIERARCHY_ARGS+=(--merge_keep_label "${MERGE_KEEP_LABEL}")
fi
if [[ -n "${MERGE_DROP_LABEL}" ]]; then
  HIERARCHY_ARGS+=(--merge_drop_label "${MERGE_DROP_LABEL}")
fi
if [[ -n "${REMOVE_SOURCE_LABEL}" ]]; then
  HIERARCHY_ARGS+=(--remove_source_label "${REMOVE_SOURCE_LABEL}")
fi
if [[ -n "${ADD_SOURCE_LABEL}" ]]; then
  HIERARCHY_ARGS+=(--add_source_label "${ADD_SOURCE_LABEL}")
fi
if [[ -n "${ADD_TARGET_LABELS}" ]]; then
  HIERARCHY_ARGS+=(--add_target_labels "${ADD_TARGET_LABELS}")
fi

python src/scripts/prepare_newsgroups_gt_hierarchies.py "${HIERARCHY_ARGS[@]}"

echo "Prep complete:"
echo "  ${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"
echo "  ${EXP_DIR}/hierarchy_results__clean"
echo "  ${EXP_DIR}/hierarchy_results__merge_only"
echo "  ${EXP_DIR}/hierarchy_results__split_only"
echo "  ${EXP_DIR}/hierarchy_results__remove_only"
echo "  ${EXP_DIR}/hierarchy_results__add_only"
