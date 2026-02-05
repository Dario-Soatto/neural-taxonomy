#!/bin/bash
#SBATCH --job-name=step2_vllm_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step2_vllm_%j.out
#SBATCH --error=logs/step2_vllm_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_GLOB="experiments/wiki_biographies_10000/wiki_biographies-initial-labeling-labeling__experiment-biographies__model_meta-llama-Meta-Llama-3.1-8B-Instruct__*.json"
OUTPUT_FILE="experiments/wiki_biographies_10000/vllm-similarity-data.jsonl"
TEMP_DIR="experiments/wiki_biographies_10000/temp_batches"

cd "${REPO_DIR}"
mkdir -p logs "${TEMP_DIR}"

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

# Set your token in the environment before sbatch, or uncomment to hardcode.
# export HF_TOKEN="hf_..."

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
export VLLM_GUIDED_DECODING_BACKEND="outlines"

# If vLLM ignores the env flags, force its default path off /sailhome
export HOME="$LOCAL_SCRATCH"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

# Ensure guided decoding backend is available
python3 - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("outlines") else 1)
PY
if [ $? -ne 0 ]; then
  pip install outlines
fi

python src/step_2__create_supervised_similarity_data.py \
  --input_file "${INPUT_GLOB}" \
  --output_file "${OUTPUT_FILE}" \
  --model_name "${MODEL_NAME}" \
  --batch_size 5000 \
  --prompt_template biographies \
  --min_sim_threshold 0.2 \
  --max_sim_threshold 0.9 \
  --text_col_name label \
  --text_col_name_2 description \
  # --sample_size 200000 \
  --k 5 \
  --temp_dir "${TEMP_DIR}"
