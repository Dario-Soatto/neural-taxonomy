#!/bin/bash
# Quick EM grid-search launcher (fast profile, timestamped output root).
# Usage:
#   sbatch src/sbatch/em_grid_fast.sh
# Optional env overrides:
#   EM_GRID_OPERATION=split|merge|remove|add|revise
#   EM_GRID_PROFILE=fast
#   EM_GRID_BASE_DIR=experiments/wiki_biographies_10000/em_grid_search_fast
#   EM_NUM_TRIALS=2
#   EM_NUM_ITERATIONS=3
#
# Output layout example:
#   experiments/wiki_biographies_10000/em_grid_search_fast/20260227_153012_j14699999/split_fast/...

#SBATCH --job-name=em_grid_fast
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-0
#SBATCH --output=logs/em_grid_fast_%j.out
#SBATCH --error=logs/em_grid_fast_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"

EM_GRID_OPERATION="${EM_GRID_OPERATION:-split}"
EM_GRID_PROFILE="${EM_GRID_PROFILE:-fast}"
EM_NUM_TRIALS="${EM_NUM_TRIALS:-2}"
EM_NUM_ITERATIONS="${EM_NUM_ITERATIONS:-3}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-manual}"
EM_GRID_BASE_DIR="${EM_GRID_BASE_DIR:-${EXPERIMENT_DIR}/em_grid_search_fast}"
EM_GRID_OUTPUT_ROOT="${EM_GRID_BASE_DIR}/${RUN_TAG}"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer node-local SSD if available to reduce load on shared filesystems.
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/tdalmia
else
  LOCAL_SCRATCH=/nlp/scr/tdalmia
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

# Avoid /sailhome quota by putting caches on local scratch.
export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/cache/torch_extensions"
export PIP_CACHE_DIR="$LOCAL_SCRATCH/cache/pip"

# Disable vLLM usage stats writes in home cache.
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="$LOCAL_SCRATCH/cache/vllm/usage_stats.json"
export HOME="$LOCAL_SCRATCH"
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "Run tag: ${RUN_TAG}"
echo "Operation: ${EM_GRID_OPERATION}"
echo "Profile: ${EM_GRID_PROFILE}"
echo "Output root: ${EM_GRID_OUTPUT_ROOT}"

python src/em_threshold_grid_search.py \
  --operation "${EM_GRID_OPERATION}" \
  --grid_profile "${EM_GRID_PROFILE}" \
  --output_root "${EM_GRID_OUTPUT_ROOT}/${EM_GRID_OPERATION}_${EM_GRID_PROFILE}" \
  -- \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --sentence_data_path "${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" \
  --num_agglomerative_clusters 10 \
  --num_datapoints_per_cluster 50 \
  --subsample_seed 13 \
  --model_type vllm \
  --model_name "${MODEL_NAME}" \
  --sentence_column_name sentence_text \
  --log_sentence_samples 3 \
  --num_trials "${EM_NUM_TRIALS}" \
  --scorer_type batch \
  --num_iterations "${EM_NUM_ITERATIONS}" \
  --accept_metric assigned \
  --split_max_per_iter 0 \
  --split_cooldown_iters 0 \
  --merge_max_per_iter 0 \
  --remove_max_per_iter 0 \
  --revise_max_per_iter 0 \
  --revise_cooldown_iters 0 \
  --add_max_new_clusters_per_iter 50 \
  --add_cooldown_iters 0 \
  --noop_patience 1 \
  --log_iteration_metrics \
  --log_top_pairs 5 \
  --diagnostics_sample_size 50 \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2
