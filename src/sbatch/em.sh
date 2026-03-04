#!/bin/bash
#SBATCH --job-name=em_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[28-39]
#SBATCH --exclude=jagupard18,jagupard19,jagupard20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16-0
#SBATCH --output=logs/em_%j.out
#SBATCH --error=logs/em_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
EM_TUNE_OPERATION="${EM_TUNE_OPERATION:-all}"
EM_NUM_ITERATIONS="${EM_NUM_ITERATIONS:-5}"
EM_ACCEPT_METRIC="${EM_ACCEPT_METRIC:-assigned}"
EM_ACCEPT_MIN_DELTA="${EM_ACCEPT_MIN_DELTA:-0.0}"
EM_BEST_OPERATION_PER_ITERATION="${EM_BEST_OPERATION_PER_ITERATION:-true}"
EM_MAX_SAME_OPERATION_STREAK="${EM_MAX_SAME_OPERATION_STREAK:-2}"
EM_STRUCTURAL_LABEL_MODE="${EM_STRUCTURAL_LABEL_MODE:-semantic}"
EM_SPLIT_LABEL_PAIR_ATTEMPTS="${EM_SPLIT_LABEL_PAIR_ATTEMPTS:-3}"
EM_LLM_LABEL_MAX_ATTEMPTS="${EM_LLM_LABEL_MAX_ATTEMPTS:-6}"

# Structural thresholds (derived from structural sweep winners; override via --export if desired).
EM_SPLIT_LL_MARGIN="${EM_SPLIT_LL_MARGIN:-4.0}"
EM_SPLIT_CONFIDENCE_MAX="${EM_SPLIT_CONFIDENCE_MAX:-0.32}"
EM_SPLIT_GAP_MEDIAN_MIN="${EM_SPLIT_GAP_MEDIAN_MIN:-0.16}"
EM_SPLIT_MIN_CLUSTER_SIZE="${EM_SPLIT_MIN_CLUSTER_SIZE:-25}"
EM_SPLIT_MIN_CONDITIONS="${EM_SPLIT_MIN_CONDITIONS:-2}"

EM_MERGE_SIMILARITY_MIN="${EM_MERGE_SIMILARITY_MIN:-0.65}"
EM_MERGE_L_DIFF_RATIO_MAX="${EM_MERGE_L_DIFF_RATIO_MAX:-0.12}"
EM_MERGE_C_DIFF_RATIO_MAX="${EM_MERGE_C_DIFF_RATIO_MAX:-0.15}"
EM_MERGE_MIN_CONDITIONS="${EM_MERGE_MIN_CONDITIONS:-2}"

EM_REMOVE_MIN_CLUSTER_SIZE="${EM_REMOVE_MIN_CLUSTER_SIZE:-8}"
EM_REMOVE_LL_FACTOR="${EM_REMOVE_LL_FACTOR:-0.65}"

EM_ADD_LOW_CONFIDENCE_MAX="${EM_ADD_LOW_CONFIDENCE_MAX:-0.30}"
EM_ADD_LOW_CONFIDENCE_QUANTILE="${EM_ADD_LOW_CONFIDENCE_QUANTILE:-0.15}"
EM_ADD_MIN_POORLY_EXPLAINED="${EM_ADD_MIN_POORLY_EXPLAINED:-8}"
EM_ADD_ITEMS_PER_NEW_CLUSTER="${EM_ADD_ITEMS_PER_NEW_CLUSTER:-20}"
EM_ADD_MIN_GROUP_SIZE="${EM_ADD_MIN_GROUP_SIZE:-2}"
EM_ADD_ENTROPY_MIN="${EM_ADD_ENTROPY_MIN:-0.75}"
EM_ADD_COHESION_MIN="${EM_ADD_COHESION_MIN:-0.05}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${EXPERIMENT_DIR}/em_runs_structural/${RUN_TAG}"
OUTPUT_FILE="${OUTPUT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${OUTPUT_DIR}/em_diagnostics"

if [[ "${EM_BEST_OPERATION_PER_ITERATION}" == "true" || "${EM_BEST_OPERATION_PER_ITERATION}" == "1" ]]; then
  BEST_OP_FLAG="--best_operation_per_iteration"
else
  BEST_OP_FLAG="--no-best_operation_per_iteration"
fi

cd "${REPO_DIR}"
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/tdalmia
else
  LOCAL_SCRATCH=/nlp/scr/tdalmia
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

# Avoid /sailhome quota by putting caches on local scratch
export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"

# Redirect other common caches off /sailhome
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/cache/torch_extensions"
export PIP_CACHE_DIR="$LOCAL_SCRATCH/cache/pip"

# Disable vLLM usage stats to avoid writes to home cache
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="$LOCAL_SCRATCH/cache/vllm/usage_stats.json"

# If vLLM ignores the env flags, force its default path off /sailhome
export HOME="$LOCAL_SCRATCH"

# Ensure max vLLM seq length is honored
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

print_mem_snapshot() {
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') memory snapshot ====="
  echo "-- Host memory --"
  free -h || true
  echo "-- GPU memory --"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  echo "=============================================="
}

# Periodic memory snapshots for OOM diagnosis.
(
  while true; do
    print_mem_snapshot
    sleep 120
  done
) &
MEM_MONITOR_PID=$!
cleanup() {
  kill "${MEM_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

print_mem_snapshot

python src/run_em_algorithm.py \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --sentence_data_path "${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" \
  --output_file "${OUTPUT_FILE}" \
  --diagnostics_dir "${DIAGNOSTICS_DIR}" \
  --num_agglomerative_clusters 20 \
  --num_datapoints_per_cluster 100 \
  --subsample_seed 13 \
  --model_type vllm \
  --model_name "${MODEL_NAME}" \
  --sentence_column_name sentence_text \
  --log_sentence_samples 3 \
  --num_trials 3 \
  --scorer_type batch \
  --num_iterations "${EM_NUM_ITERATIONS}" \
  --tune_operation "${EM_TUNE_OPERATION}" \
  --accept_metric "${EM_ACCEPT_METRIC}" \
  --accept_min_delta "${EM_ACCEPT_MIN_DELTA}" \
  --structural_label_mode "${EM_STRUCTURAL_LABEL_MODE}" \
  --split_label_pair_attempts "${EM_SPLIT_LABEL_PAIR_ATTEMPTS}" \
  --llm_label_max_attempts "${EM_LLM_LABEL_MAX_ATTEMPTS}" \
  ${BEST_OP_FLAG} \
  --max_same_operation_streak "${EM_MAX_SAME_OPERATION_STREAK}" \
  --split_ll_margin "${EM_SPLIT_LL_MARGIN}" \
  --split_confidence_max "${EM_SPLIT_CONFIDENCE_MAX}" \
  --split_gap_median_min "${EM_SPLIT_GAP_MEDIAN_MIN}" \
  --split_min_cluster_size "${EM_SPLIT_MIN_CLUSTER_SIZE}" \
  --split_min_conditions "${EM_SPLIT_MIN_CONDITIONS}" \
  --split_max_per_iter 0 \
  --split_cooldown_iters 0 \
  --merge_similarity_min "${EM_MERGE_SIMILARITY_MIN}" \
  --merge_l_diff_ratio_max "${EM_MERGE_L_DIFF_RATIO_MAX}" \
  --merge_c_diff_ratio_max "${EM_MERGE_C_DIFF_RATIO_MAX}" \
  --merge_min_conditions "${EM_MERGE_MIN_CONDITIONS}" \
  --merge_max_per_iter 0 \
  --remove_min_cluster_size "${EM_REMOVE_MIN_CLUSTER_SIZE}" \
  --remove_ll_factor "${EM_REMOVE_LL_FACTOR}" \
  --remove_max_per_iter 0 \
  --no-revise_enabled \
  --revise_max_per_iter 0 \
  --revise_cooldown_iters 0 \
  --add_low_confidence_max "${EM_ADD_LOW_CONFIDENCE_MAX}" \
  --add_low_confidence_quantile "${EM_ADD_LOW_CONFIDENCE_QUANTILE}" \
  --add_min_poorly_explained "${EM_ADD_MIN_POORLY_EXPLAINED}" \
  --add_items_per_new_cluster "${EM_ADD_ITEMS_PER_NEW_CLUSTER}" \
  --add_min_group_size "${EM_ADD_MIN_GROUP_SIZE}" \
  --add_entropy_min "${EM_ADD_ENTROPY_MIN}" \
  --add_cohesion_min "${EM_ADD_COHESION_MIN}" \
  --add_max_new_clusters_per_iter 50 \
  --add_cooldown_iters 0 \
  --noop_patience 1 \
  --log_iteration_metrics \
  --log_top_pairs 5 \
  --diagnostics_sample_size 50 \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2

echo "Structural all-ops run completed."
echo "Output directory: ${OUTPUT_DIR}"
echo "Scores CSV: ${OUTPUT_FILE}"
echo "Diagnostics: ${DIAGNOSTICS_DIR}"

print_mem_snapshot
