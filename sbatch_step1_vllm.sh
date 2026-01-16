#!/bin/bash
#SBATCH --job-name=step1_vllm_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-lo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step1_vllm_%j.out
#SBATCH --error=logs/step1_vllm_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
INPUT_FILE="experiments/wiki_biographies/data/processed_custom_input_subset.csv"
OUTPUT_FILE="experiments/wiki_biographies/wiki_biographies-initial-labeling.json"

cd "${REPO_DIR}"
mkdir -p logs

# Avoid /sailhome quota by putting HF caches on scratch
export HF_HOME=/nlp/scr/tdalmia/hf_cache
export HUGGINGFACE_HUB_CACHE=/nlp/scr/tdalmia/hf_cache/hub
export TRANSFORMERS_CACHE=/nlp/scr/tdalmia/hf_cache/transformers
export HF_DATASETS_CACHE=/nlp/scr/tdalmia/hf_cache/datasets
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

# Redirect flashinfer JIT cache off /sailhome
export XDG_CACHE_HOME=/nlp/scr/tdalmia/.cache
export FLASHINFER_WORKSPACE_DIR=/nlp/scr/tdalmia/.cache/flashinfer
mkdir -p "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_DIR"

# Redirect other common caches off /sailhome
export TORCH_HOME=/nlp/scr/tdalmia/.cache/torch
export TORCH_EXTENSIONS_DIR=/nlp/scr/tdalmia/.cache/torch_extensions
export PIP_CACHE_DIR=/nlp/scr/tdalmia/.cache/pip
mkdir -p "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$PIP_CACHE_DIR"

# Disable vLLM usage stats to avoid writes to home cache
export VLLM_USAGE_STATS=0
export VLLM_USAGE_STATS_PATH=/nlp/scr/tdalmia/.cache/vllm/usage_stats.json
mkdir -p /nlp/scr/tdalmia/.cache/vllm

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

python src/step_1__run_initial_labeling_prompts.py \
  --model "${MODEL_NAME}" \
  --input_data_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --experiment editorials \
  --start_idx 0 \
  --end_idx 40 \
  --batch_size 4 \
  --num_sents_per_prompt 1
