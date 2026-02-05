#!/bin/bash
#SBATCH --job-name=step1_vllm_wiki
#SBATCH --account=nlp
#SBATCH --nodelist=jagupard29
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step1_vllm_%j.out
#SBATCH --error=logs/step1_vllm_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_FILE="experiments/wiki_biographies/data/processed_custom_input_subset_10000.csv"
OUTPUT_FILE="experiments/wiki_biographies_10000/wiki_biographies-initial-labeling.json"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/tdalmia
else
  LOCAL_SCRATCH=/nlp/scr/tdalmia
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/flashinfer,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

# Avoid /sailhome quota by putting caches on local scratch
export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"

# Redirect flashinfer JIT cache off /sailhome
export XDG_CACHE_HOME="$LOCAL_SCRATCH/cache"
export FLASHINFER_WORKSPACE_DIR="$LOCAL_SCRATCH/cache/flashinfer"

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

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

python src/step_1__run_initial_labeling_prompts.py \
  --model "${MODEL_NAME}" \
  --input_data_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --experiment biographies \
  --start_idx 0 \
  --end_idx 10000 \
  --batch_size 16 \
  --num_sents_per_prompt 1 \
  --overwrite_prompt_cache \
  --window_size 3 
