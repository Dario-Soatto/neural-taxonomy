#!/bin/bash
# Score all baseline outputs with LLM perplexity.
# Requires GPU for vLLM model loading.
#
# Usage:
#   sbatch src/sbatch/baselines_llm_score.sh
#   sbatch --export=ALL,DATASET=wiki src/sbatch/baselines_llm_score.sh

#SBATCH --job-name=baselines_llm
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[28-39]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=logs/baselines_llm_%j.out
#SBATCH --error=logs/baselines_llm_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
DATASET="${DATASET:-newsgroups}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
NUM_TRIALS="${NUM_TRIALS:-3}"

cd "${REPO_DIR}"
mkdir -p logs

# Setup caches
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/vllm}

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export HOME="$LOCAL_SCRATCH"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "========================================"
echo "Baseline LLM scoring"
echo "Started:     $(date)"
echo "Dataset:     ${DATASET}"
echo "Model:       ${MODEL_NAME}"
echo "Num trials:  ${NUM_TRIALS}"
echo "========================================"

score_dataset() {
  local EXP_DIR="$1"
  local TEXT_COL="$2"
  local BASELINE_ROOT="${EXP_DIR}/baseline_results"

  if [[ ! -d "${BASELINE_ROOT}" ]]; then
    echo "No baseline results found at ${BASELINE_ROOT}. Run baselines_cpu.sh first."
    return 1
  fi

  # Collect all baseline directories (skip sub-run dirs, handle both flat and multi-run layouts)
  BASELINE_DIRS=()
  for method_dir in "${BASELINE_ROOT}"/*/; do
    method_name=$(basename "${method_dir}")
    # Check if this dir contains sub-runs
    if ls "${method_dir}"/run_*/inner_node_labels.csv >/dev/null 2>&1; then
      for run_dir in "${method_dir}"/run_*/; do
        BASELINE_DIRS+=("${run_dir%/}")
      done
    elif [[ -f "${method_dir}/inner_node_labels.csv" ]]; then
      BASELINE_DIRS+=("${method_dir%/}")
    fi
  done

  if [[ ${#BASELINE_DIRS[@]} -eq 0 ]]; then
    echo "No baseline directories with inner_node_labels.csv found."
    return 1
  fi

  echo "Scoring ${#BASELINE_DIRS[@]} baseline directories..."

  python src/score_baseline_with_llm.py \
    --baseline_dirs "${BASELINE_DIRS[@]}" \
    --sentence_data_path "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv" \
    --sentence_column "${TEXT_COL}" \
    --model_name "${MODEL_NAME}" \
    --model_type vllm \
    --num_trials "${NUM_TRIALS}"
}

if [[ "${DATASET}" == "newsgroups" || "${DATASET}" == "both" ]]; then
  score_dataset "experiments/newsgroups_sanity_tiny" "description"
fi

if [[ "${DATASET}" == "wiki" || "${DATASET}" == "both" ]]; then
  score_dataset "experiments/wiki_biographies_10000" "sentence_text"
fi

echo "========================================"
echo "Finished:    $(date)"
echo "========================================"
