#!/bin/bash
#SBATCH --job-name=bbc_llm_hier
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bbc_news_llm_hierarchy_%j.out
#SBATCH --error=logs/bbc_news_llm_hierarchy_%j.err

###############################################################################
# BBC News: iterative LLM hierarchy (run_llm_iterative_hierarchy.py) + eval.
#
# VRAM: Llama-3.1-8B-Instruct in fp16 needs roughly >=16GB GPU for vLLM. On a
# ~12GB card vLLM will OOM at load. Options:
#   - Request a larger GPU, e.g.  sbatch --gres=gpu:a6000:1  src/sbatch/...
#   - Or use the API:  export USE_OPENAI=1 OPENAI_API_KEY=...  (see below)
# Optional:  export VLLM_GPU_MEMORY_UTILIZATION=0.80  (helps only on borderline VRAM)
#
# Prerequisites (run once): Steps 1–3 of the BBC pipeline must exist under
#   experiments/bbc_news/
# i.e. Step 1 labeling JSON shard + triplets + trained SBERT at
#   experiments/bbc_news/models/bbc_news-sentence-similarity-model/trained-model
#
# Full prerequisite pipeline:
#   sbatch src/sbatch/run_bbc_news.sh   # stop after Step 3, OR let it finish
#
# This job does NOT run Steps 4–6 (classic hierarchy); it writes to a separate
# experiment directory so nothing is overwritten.
#
# Usage (from repo root):
#   mkdir -p logs
#   sbatch src/sbatch/run_bbc_news_llm_iterative_hierarchy.sh
#
# OpenAI instead of vLLM on the node:
#   export OPENAI_API_KEY="sk-..."
#   sbatch --export=ALL,OPENAI_API_KEY src/sbatch/run_bbc_news_llm_iterative_hierarchy.sh
# and set USE_OPENAI=1 below or export USE_OPENAI=1 before sbatch.
###############################################################################

set -euo pipefail

USE_OPENAI="${USE_OPENAI:-0}"

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

EXPERIMENT_NAME="bbc_news"
EXPERIMENT_DIR="${REPO_DIR}/experiments/${EXPERIMENT_NAME}"
OUTPUT_DIR="${REPO_DIR}/experiments/bbc_news_llm_iterative"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
SBERT_TRAINED="${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}-sentence-similarity-model/trained-model"
INPUT_CSV="${EXPERIMENT_DIR}/data/bbc_news.csv"

NCENTROIDS="${NCENTROIDS:-1024}"
K_BATCH="${K_BATCH:-8}"
SEED="${SEED:-42}"

cd "${REPO_DIR}"
echo "REPO_DIR=${REPO_DIR}"
echo "GIT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
mkdir -p logs "${OUTPUT_DIR}"

# ── Same cache layout as run_bbc_news.sh ─────────────────────────────────────
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/flashinfer,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export XDG_CACHE_HOME="$LOCAL_SCRATCH/cache"
export FLASHINFER_WORKSPACE_DIR="$LOCAL_SCRATCH/cache/flashinfer"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/cache/torch_extensions"
export PIP_CACHE_DIR="$LOCAL_SCRATCH/cache/pip"
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="$LOCAL_SCRATCH/cache/vllm/usage_stats.json"

if [ -z "${HF_TOKEN:-}" ]; then
    for _tok_path in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        if [ -f "${_tok_path}" ]; then
            export HF_TOKEN="$(cat "${_tok_path}")"
            break
        fi
    done
fi
export HOME="$LOCAL_SCRATCH"
# Must fit prompt (full hierarchy grows) + completion. 8192 causes truncated JSON.
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

source /nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "========================================"
echo "BBC LLM iterative hierarchy: $(date)"
echo "MODEL_NAME=${MODEL_NAME}"
echo "USE_OPENAI=${USE_OPENAI}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "========================================"

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "ERROR: Missing ${INPUT_CSV}. Run prepare_bbc_news_data.py or run_bbc_news.sh Step 0 first." >&2
  exit 1
fi

if [[ ! -d "${SBERT_TRAINED}" ]]; then
  echo "ERROR: Trained SBERT not found at ${SBERT_TRAINED}." >&2
  echo "Run Steps 1–3 (e.g. sbatch src/sbatch/run_bbc_news.sh and wait for Step 3)." >&2
  exit 1
fi

STEP1_RESULT=$(ls "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling-labeling__experiment-bbc-news__model_${MODEL_NAME//\//-}__"*.json 2>/dev/null | head -n 1)
if [[ -z "${STEP1_RESULT}" ]]; then
  echo "ERROR: No Step 1 JSON matching model ${MODEL_NAME} under ${EXPERIMENT_DIR}." >&2
  echo "Run Step 1 with this model or set MODEL_NAME to match your existing shard." >&2
  exit 1
fi
echo "Step 1 input: ${STEP1_RESULT}"

LLM_EXTRA_ARGS=()
if [[ "${USE_OPENAI}" == "1" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: USE_OPENAI=1 but OPENAI_API_KEY is not set. Use: sbatch --export=ALL,OPENAI_API_KEY ..." >&2
    exit 1
  fi
  LLM_EXTRA_ARGS+=(--use_openai)
fi

echo ">>> Running iterative LLM hierarchy..."
python src/run_llm_iterative_hierarchy.py \
  --input_data_file "${STEP1_RESULT}" \
  --trained_sbert_model_name "${SBERT_TRAINED}" \
  --output_dir "${OUTPUT_DIR}" \
  --input_col_name label \
  --label_col label \
  --text_col description \
  --experiment bbc-news \
  --ncentroids "${NCENTROIDS}" \
  --k_batch "${K_BATCH}" \
  --seed "${SEED}" \
  --model_name "${MODEL_NAME}" \
  --max_tokens 16384 \
  "${LLM_EXTRA_ARGS[@]}"

echo ">>> Evaluating (same as evaluate_pipeline.py)..."
python src/scripts/evaluate_pipeline.py \
  --experiment_dir "${OUTPUT_DIR}" \
  --raw_data_csv "${INPUT_CSV}" \
  --true_label_col true_label

echo "========================================"
echo "Done: $(date)"
echo "Results: ${OUTPUT_DIR}/evaluation_results.csv"
echo "========================================"
