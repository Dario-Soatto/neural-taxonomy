#!/bin/bash
#SBATCH --job-name=em_replicate
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16-0
#SBATCH --output=logs/em_%j.out
#SBATCH --error=logs/em_%j.err

###############################################################################
# Replicates Tushar's best EM run (job 14713446, March 3 2026).
#
# Known config from the .err log:
#   - experiment_dir: experiments/wiki_biographies_10000
#   - Level 2, N_Clusters 10 => --num_agglomerative_clusters 10
#   - Subsampling to 50 datapoints per cluster, seed=13
#   - model_type: vllm, model: meta-llama/Meta-Llama-3.1-8B-Instruct
#   - scorer: batch, permutations: True, prompts: False => --num_trials 3
#   - tune_operation=all | split=T merge=T remove=T add=T revise=F
#   - best_operation_per_iteration=True, max_same_operation_streak=2
#   - structural_label_mode=semantic, split_label_pair_attempts=3
#   - accept_metric=assigned, accept_min_delta=0.0
#   - Structural thresholds: em.sh defaults ("structural sweep winners")
#
# Uses the ORIGINAL run_em_algorithm.py (sentence-transformer + LLM).
###############################################################################

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="em_runs_structural_replicate/${RUN_TAG}"
OUTPUT_FILE="${OUTPUT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${OUTPUT_DIR}/em_diagnostics"

cd "${REPO_DIR}"
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/soatto
else
  LOCAL_SCRATCH=/nlp/scr/soatto
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

source /nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh
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
  --num_agglomerative_clusters 10 \
  --num_datapoints_per_cluster 50 \
  --subsample_seed 13 \
  --model_type vllm \
  --model_name "${MODEL_NAME}" \
  --sentence_column_name sentence_text \
  --log_sentence_samples 3 \
  --num_trials 3 \
  --scorer_type batch \
  --num_iterations 5 \
  --tune_operation all \
  --accept_metric assigned \
  --accept_min_delta 0.0 \
  --structural_label_mode semantic \
  --split_label_pair_attempts 3 \
  --llm_label_max_attempts 6 \
  --best_operation_per_iteration \
  --max_same_operation_streak 2 \
  --split_enabled \
  --split_ll_margin 4.0 \
  --split_confidence_max 0.32 \
  --split_gap_median_min 0.16 \
  --split_min_cluster_size 25 \
  --split_min_conditions 2 \
  --split_max_per_iter 0 \
  --split_cooldown_iters 0 \
  --merge_enabled \
  --merge_similarity_min 0.65 \
  --merge_l_diff_ratio_max 0.12 \
  --merge_c_diff_ratio_max 0.15 \
  --merge_min_conditions 2 \
  --merge_max_per_iter 0 \
  --remove_enabled \
  --remove_min_cluster_size 8 \
  --remove_ll_factor 0.65 \
  --remove_max_per_iter 0 \
  --no-revise_enabled \
  --revise_max_per_iter 0 \
  --revise_cooldown_iters 0 \
  --add_enabled \
  --add_low_confidence_max 0.30 \
  --add_low_confidence_quantile 0.15 \
  --add_min_poorly_explained 8 \
  --add_items_per_new_cluster 20 \
  --add_min_group_size 2 \
  --add_entropy_min 0.75 \
  --add_cohesion_min 0.05 \
  --add_max_new_clusters_per_iter 50 \
  --add_cooldown_iters 0 \
  --noop_patience 1 \
  --log_iteration_metrics \
  --log_top_pairs 5 \
  --diagnostics_sample_size 50 \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2

echo "Replicate best run (original EM) completed."
echo "Output directory: ${OUTPUT_DIR}"
echo "Scores CSV: ${OUTPUT_FILE}"
echo "Diagnostics: ${DIAGNOSTICS_DIR}"

print_mem_snapshot
