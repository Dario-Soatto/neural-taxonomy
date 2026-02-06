#!/bin/bash
#SBATCH --job-name=em_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=14-0
#SBATCH --output=logs/em_%j.out
#SBATCH --error=logs/em_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies"
OUTPUT_FILE="experiments/wiki_biographies/em_refined_scores.csv"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

cd "${REPO_DIR}"
mkdir -p logs

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
  --output_file "${OUTPUT_FILE}" \
  --num_agglomerative_clusters 48 \
  --model_type vllm \
  --model_name "${MODEL_NAME}" \
  --sentence_column_name sentences \
  --num_trials 3 \
  --scorer_type batch \
  --num_iterations 3 \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --num_datapoints_per_cluster 50 \
  --show_progress \
  --log_iteration_metrics \
  --log_top_pairs 10 \
  --diagnostics_sample_size 200 \
  --diagnostics_bins 40

print_mem_snapshot