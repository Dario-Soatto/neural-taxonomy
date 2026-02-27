#!/bin/bash
#SBATCH --job-name=em_wiki_grid
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[28-39]
#SBATCH --exclude=jagupard18,jagupard19,jagupard20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=14-0
#SBATCH --output=logs/em_grid_%j.out
#SBATCH --error=logs/em_grid_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_10000"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
EM_GRID_OPERATION="${EM_GRID_OPERATION:-add}"
EM_GRID_PROFILE="${EM_GRID_PROFILE:-coarse}"
EM_GRID_OUTPUT_ROOT="${EM_GRID_OUTPUT_ROOT:-${EXPERIMENT_DIR}/em_grid_search}"

cd "${REPO_DIR}"
mkdir -p logs

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
  --num_trials 3 \
  --scorer_type batch \
  --num_iterations 5 \
  --split_max_per_iter 0 \
  --split_cooldown_iters 0 \
  --merge_max_per_iter 0 \
  --remove_max_per_iter 0 \
  --revise_max_per_iter 0 \
  --revise_cooldown_iters 0 \
  --add_max_new_clusters_per_iter 50 \
  --add_cooldown_iters 0 \
  --noop_patience 1 \
  --log_top_pairs 5 \
  --diagnostics_sample_size 50 \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2
