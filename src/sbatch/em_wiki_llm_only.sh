#!/bin/bash
# LLM-only EM run for wiki biographies.
#
# Default usage:
#   sbatch src/sbatch/em_wiki_llm_only_quick.sh
#
# Override small knobs at submit time:
#   sbatch --export=ALL,N_CLUSTERS=10,POINTS_PER_CLUSTER=25,N_ITERS=2 src/sbatch/em_wiki_llm_only_quick.sh
#   sbatch --export=ALL,N_CLUSTERS=8,POINTS_PER_CLUSTER=15,N_ITERS=1,MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct src/sbatch/em_wiki_llm_only_quick.sh

#SBATCH --job-name=em_wiki_llm_quick
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/em_wiki_llm_%j.out
#SBATCH --error=logs/em_wiki_llm_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-experiments/wiki_biographies_10000}"
SENTENCE_DATA_PATH="${SENTENCE_DATA_PATH:-${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-sentence-transformers/all-MiniLM-L6-v2}"

N_CLUSTERS="${N_CLUSTERS:-10}"
POINTS_PER_CLUSTER="${POINTS_PER_CLUSTER:-25}"
N_ITERS="${N_ITERS:-3}"
SUBSAMPLE_SEED="${SUBSAMPLE_SEED:-13}"
NUM_TRIALS="${NUM_TRIALS:-3}"

SENTENCE_COLUMN_NAME="${SENTENCE_COLUMN_NAME:-sentence_text}"
TUNE_OPERATION="${TUNE_OPERATION:-all}"
STRUCTURAL_LABEL_MODE="${STRUCTURAL_LABEL_MODE:-semantic}"
LABEL_PROMPT_STYLE="${LABEL_PROMPT_STYLE:-auto}"
ACCEPT_METRIC="${ACCEPT_METRIC:-assigned}"
ACCEPT_MIN_DELTA="${ACCEPT_MIN_DELTA:-0.0005}"
NOOP_PATIENCE="${NOOP_PATIENCE:-3}"
PROPOSAL_TOP_K_LOG="${PROPOSAL_TOP_K_LOG:-5}"
PROPORTIONAL_THRESHOLDS="${PROPORTIONAL_THRESHOLDS:-1}"
ADAPTIVE_PROPOSAL_THRESHOLDS="${ADAPTIVE_PROPOSAL_THRESHOLDS:-1}"
ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS="${ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS:-3}"
DIAGNOSTICS_SAMPLE_SIZE="${DIAGNOSTICS_SAMPLE_SIZE:-50}"
LOG_SENTENCE_SAMPLES="${LOG_SENTENCE_SAMPLES:-3}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/em_runs_llm_only_quick/${RUN_TAG}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/em_refined_scores.csv}"
DIAGNOSTICS_DIR="${DIAGNOSTICS_DIR:-${OUTPUT_DIR}/em_diagnostics}"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}" "${DIAGNOSTICS_DIR}"

if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/cache/torch_extensions"
export PIP_CACHE_DIR="$LOCAL_SCRATCH/cache/pip"
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="$LOCAL_SCRATCH/cache/vllm/usage_stats.json"
export HOME="$LOCAL_SCRATCH"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "========================================"
echo "Quick wiki biographies LLM-only EM run"
echo "Started:             $(date)"
echo "Repo:                ${REPO_DIR}"
echo "Experiment:          ${EXPERIMENT_DIR}"
echo "Sentence data:       ${SENTENCE_DATA_PATH}"
echo "Model:               ${MODEL_NAME}"
echo "Clusters:            ${N_CLUSTERS}"
echo "Points per cluster:  ${POINTS_PER_CLUSTER}"
echo "Iterations:          ${N_ITERS}"
echo "Sentence column:     ${SENTENCE_COLUMN_NAME}"
echo "Tune operation:      ${TUNE_OPERATION}"
echo "Label prompt style:  ${LABEL_PROMPT_STYLE}"
echo "Proportional props:  ${PROPORTIONAL_THRESHOLDS}"
echo "Adaptive props:      ${ADAPTIVE_PROPOSAL_THRESHOLDS}"
echo "Adaptive max relax:  ${ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS}"
echo "Output dir:          ${OUTPUT_DIR}"
echo "========================================"

EM_ARGS=(
  --experiment_dir "${EXPERIMENT_DIR}"
  --sentence_data_path "${SENTENCE_DATA_PATH}"
  --output_file "${OUTPUT_FILE}"
  --diagnostics_dir "${DIAGNOSTICS_DIR}"
  --num_agglomerative_clusters "${N_CLUSTERS}"
  --num_datapoints_per_cluster "${POINTS_PER_CLUSTER}"
  --subsample_seed "${SUBSAMPLE_SEED}"
  --model_type vllm
  --model_name "${MODEL_NAME}"
  --sentence_column_name "${SENTENCE_COLUMN_NAME}"
  --log_sentence_samples "${LOG_SENTENCE_SAMPLES}"
  --num_trials "${NUM_TRIALS}"
  --scorer_type batch
  --num_iterations "${N_ITERS}"
  --tune_operation "${TUNE_OPERATION}"
  --accept_metric "${ACCEPT_METRIC}"
  --accept_min_delta "${ACCEPT_MIN_DELTA}"
  --structural_label_mode "${STRUCTURAL_LABEL_MODE}"
  --label_prompt_style "${LABEL_PROMPT_STYLE}"
  --best_operation_per_iteration
  --noop_patience "${NOOP_PATIENCE}"
  --log_iteration_metrics
  --proposal_top_k_log "${PROPOSAL_TOP_K_LOG}"
  --log_top_pairs 5
  --diagnostics_sample_size "${DIAGNOSTICS_SAMPLE_SIZE}"
  --embedding_model_name "${EMBEDDING_MODEL_NAME}"
  --adaptive_threshold_max_relax_iters "${ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS}"
  --show_progress
)

if [[ "${PROPORTIONAL_THRESHOLDS}" == "1" ]]; then
  EM_ARGS+=(--proposal_proportional_thresholds)
else
  EM_ARGS+=(--no-proposal_proportional_thresholds)
fi

if [[ "${ADAPTIVE_PROPOSAL_THRESHOLDS}" == "1" ]]; then
  EM_ARGS+=(--proposal_adaptive_thresholds)
else
  EM_ARGS+=(--no-proposal_adaptive_thresholds)
fi

python src/run_em_algorithm_llm_only.py "${EM_ARGS[@]}"

echo "========================================"
echo "Finished:            $(date)"
echo "Scores CSV:          ${OUTPUT_FILE}"
echo "Diagnostics:         ${DIAGNOSTICS_DIR}"
echo "========================================"




