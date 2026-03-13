#!/bin/bash
#SBATCH --job-name=ng_em_tiny
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[28-39]
#SBATCH --exclude=jagupard18,jagupard19,jagupard20
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/newsgroups_em_%j.out
#SBATCH --error=logs/newsgroups_em_%j.err

set -euo pipefail

# Allow overrides at submit time:
#   sbatch --export=ALL,SANITY_MODE=clean src/sbatch/newsgroups_sanity_em_llm_only_jag.sh
REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
EXP_NAME="${EXP_NAME:-newsgroups_sanity_tiny}"
EXP_DIR="${EXP_DIR:-${REPO_DIR}/experiments/${EXP_NAME}}"

SANITY_MODE="${SANITY_MODE:-clean}"
N_CLUSTERS="${N_CLUSTERS:-8}"
POINTS_PER_CLUSTER="${POINTS_PER_CLUSTER:-}"
N_ITERS="${N_ITERS:-3}"
NUM_TRIALS="${NUM_TRIALS:-3}"
SCORER_TYPE="${SCORER_TYPE:-batch}"
SEED="${SEED:-13}"
SENTENCE_COLUMN_NAME="${SENTENCE_COLUMN_NAME:-description}"
TUNE_OPERATION="${TUNE_OPERATION:-}"
BEST_OPERATION_PER_ITER="${BEST_OPERATION_PER_ITER:-1}"
SPLIT_ENABLED="${SPLIT_ENABLED:-}"
MERGE_ENABLED="${MERGE_ENABLED:-}"
REMOVE_ENABLED="${REMOVE_ENABLED:-}"
ADD_ENABLED="${ADD_ENABLED:-}"
HIERARCHY_SUBDIR="${HIERARCHY_SUBDIR:-hierarchy_results__${SANITY_MODE}}"
HIERARCHY_DIR="${HIERARCHY_DIR:-${EXP_DIR}/${HIERARCHY_SUBDIR}}"
LOG_TOP_PAIRS="${LOG_TOP_PAIRS:-0}"
DIAGNOSTICS_SAMPLE_SIZE="${DIAGNOSTICS_SAMPLE_SIZE:-200}"
REVISE_ENABLED="${REVISE_ENABLED:-0}"
SPLIT_SIZE_GATE_MODE="${SPLIT_SIZE_GATE_MODE:-fixed}"
SPLIT_MIN_CLUSTER_SIZE_MEAN_RATIO="${SPLIT_MIN_CLUSTER_SIZE_MEAN_RATIO:-0.50}"
SPLIT_MAX_PER_ITER="${SPLIT_MAX_PER_ITER:-1}"
SPLIT_GAP_TOP2_MAX="${SPLIT_GAP_TOP2_MAX:-0.25}"
SPLIT_RUNNER_UP_MIN_SHARE="${SPLIT_RUNNER_UP_MIN_SHARE:-0.30}"
MERGE_EPSILON_MODE="${MERGE_EPSILON_MODE:-fixed}"
MERGE_L_ABS_DIFF_QUANTILE="${MERGE_L_ABS_DIFF_QUANTILE:-0.25}"
MERGE_C_ABS_DIFF_QUANTILE="${MERGE_C_ABS_DIFF_QUANTILE:-0.25}"
MERGE_SIMILARITY_MODE="${MERGE_SIMILARITY_MODE:-fixed}"
MERGE_SIMILARITY_QUANTILE="${MERGE_SIMILARITY_QUANTILE:-0.75}"
MERGE_MAX_PER_ITER="${MERGE_MAX_PER_ITER:-1}"
REMOVE_MAX_PER_ITER="${REMOVE_MAX_PER_ITER:-1}"
REMOVE_MIN_CLUSTER_SIZE_MEAN_RATIO="${REMOVE_MIN_CLUSTER_SIZE_MEAN_RATIO:-0.30}"
ADD_MAX_NEW_CLUSTERS_PER_ITER="${ADD_MAX_NEW_CLUSTERS_PER_ITER:-1}"
PROPORTIONAL_THRESHOLDS="${PROPORTIONAL_THRESHOLDS:-0}"
ADAPTIVE_PROPOSAL_THRESHOLDS="${ADAPTIVE_PROPOSAL_THRESHOLDS:-0}"
ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS="${ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS:-3}"
ADAPTIVE_QUANTILE_STEP="${ADAPTIVE_QUANTILE_STEP:-0.08}"
ADAPTIVE_SPLIT_SIZE_RELAX_RATIO="${ADAPTIVE_SPLIT_SIZE_RELAX_RATIO:-0.20}"
ADAPTIVE_MERGE_SIMILARITY_RELAX="${ADAPTIVE_MERGE_SIMILARITY_RELAX:-0.10}"
ADAPTIVE_MERGE_EPSILON_EXPAND="${ADAPTIVE_MERGE_EPSILON_EXPAND:-0.30}"
ADAPTIVE_REMOVE_SIZE_RELAX_RATIO="${ADAPTIVE_REMOVE_SIZE_RELAX_RATIO:-0.20}"
ADAPTIVE_REMOVE_LL_FACTOR_STEP="${ADAPTIVE_REMOVE_LL_FACTOR_STEP:-0.01}"
PROPOSAL_TOP_K_LOG="${PROPOSAL_TOP_K_LOG:-0}"
# Default to semantic LLM relabeling for split/merge/add/revise. Set
# STRUCTURAL_LABEL_MODE=parent_relative at submit time if you want deterministic placeholders.
STRUCTURAL_LABEL_MODE="${STRUCTURAL_LABEL_MODE:-semantic}"
LABEL_PROMPT_STYLE="${LABEL_PROMPT_STYLE:-auto}"
ACCEPT_MIN_DELTA="${ACCEPT_MIN_DELTA:-0.0005}"
NOOP_PATIENCE="${NOOP_PATIENCE:-2}"

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"

case "${SANITY_MODE}" in
  clean)
    TUNE_OPERATION="${TUNE_OPERATION:-all}"
    SPLIT_ENABLED="${SPLIT_ENABLED:-1}"
    MERGE_ENABLED="${MERGE_ENABLED:-1}"
    REMOVE_ENABLED="${REMOVE_ENABLED:-1}"
    ADD_ENABLED="${ADD_ENABLED:-1}"
    ;;
  merge_only)
    TUNE_OPERATION="${TUNE_OPERATION:-merge}"
    SPLIT_ENABLED="${SPLIT_ENABLED:-0}"
    MERGE_ENABLED="${MERGE_ENABLED:-1}"
    REMOVE_ENABLED="${REMOVE_ENABLED:-0}"
    ADD_ENABLED="${ADD_ENABLED:-0}"
    ;;
  split_only)
    TUNE_OPERATION="${TUNE_OPERATION:-split}"
    SPLIT_ENABLED="${SPLIT_ENABLED:-1}"
    MERGE_ENABLED="${MERGE_ENABLED:-0}"
    REMOVE_ENABLED="${REMOVE_ENABLED:-0}"
    ADD_ENABLED="${ADD_ENABLED:-0}"
    ;;
  remove_only)
    TUNE_OPERATION="${TUNE_OPERATION:-remove}"
    SPLIT_ENABLED="${SPLIT_ENABLED:-0}"
    MERGE_ENABLED="${MERGE_ENABLED:-0}"
    REMOVE_ENABLED="${REMOVE_ENABLED:-1}"
    ADD_ENABLED="${ADD_ENABLED:-0}"
    ;;
  add_only)
    TUNE_OPERATION="${TUNE_OPERATION:-add}"
    SPLIT_ENABLED="${SPLIT_ENABLED:-0}"
    MERGE_ENABLED="${MERGE_ENABLED:-0}"
    REMOVE_ENABLED="${REMOVE_ENABLED:-0}"
    ADD_ENABLED="${ADD_ENABLED:-1}"
    ;;
  *)
    echo "Unsupported SANITY_MODE='${SANITY_MODE}'. Expected clean, merge_only, split_only, remove_only, or add_only." >&2
    exit 2
    ;;
esac

RUN_TAG="${SANITY_MODE}_$(date +%Y%m%d_%H%M%S)_j${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${EXP_DIR}/em_runs_llm_only/${RUN_TAG}"
OUTPUT_FILE="${OUTPUT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${OUTPUT_DIR}/em_diagnostics"

cd "${REPO_DIR}"
mkdir -p logs "${OUTPUT_DIR}"

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
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
export VLLM_CHOICE_BATCH_SIZE="${VLLM_CHOICE_BATCH_SIZE:-2}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "=== Newsgroups EM tiny run (LLM-only) ==="
echo "repo:               ${REPO_DIR}"
echo "exp_dir:            ${EXP_DIR}"
echo "sanity_mode:        ${SANITY_MODE}"
echo "n_clusters:         ${N_CLUSTERS}"
echo "hierarchy_dir:      ${HIERARCHY_DIR}"
echo "points_per_cluster: ${POINTS_PER_CLUSTER:-<python_default_none>}"
echo "num_iterations:     ${N_ITERS}"
echo "num_trials:         ${NUM_TRIALS}"
echo "scorer_type:        ${SCORER_TYPE}"
echo "tune_operation:     ${TUNE_OPERATION}"
echo "best_op_per_iter:   ${BEST_OPERATION_PER_ITER}"
echo "ops enabled:        split=${SPLIT_ENABLED} merge=${MERGE_ENABLED} remove=${REMOVE_ENABLED} add=${ADD_ENABLED} revise=${REVISE_ENABLED}"
echo "revise_enabled:     ${REVISE_ENABLED}"
echo "split_size_mode:    ${SPLIT_SIZE_GATE_MODE}"
echo "split_max_per_iter: ${SPLIT_MAX_PER_ITER}"
echo "split_gap_top2_max: ${SPLIT_GAP_TOP2_MAX}"
echo "split_runner_share: ${SPLIT_RUNNER_UP_MIN_SHARE}"
echo "merge_eps_mode:     ${MERGE_EPSILON_MODE}"
echo "merge_sim_mode:     ${MERGE_SIMILARITY_MODE}"
echo "merge_max_per_iter: ${MERGE_MAX_PER_ITER}"
echo "remove_max_per_iter:${REMOVE_MAX_PER_ITER}"
echo "add_max_new:        ${ADD_MAX_NEW_CLUSTERS_PER_ITER}"
echo "proportional_props: ${PROPORTIONAL_THRESHOLDS}"
echo "adaptive_props:     ${ADAPTIVE_PROPOSAL_THRESHOLDS}"
echo "proposal_top_k_log: ${PROPOSAL_TOP_K_LOG}"
echo "sentence_column:    ${SENTENCE_COLUMN_NAME}"
echo "structural_labels:  ${STRUCTURAL_LABEL_MODE}"
echo "label_prompt_style: ${LABEL_PROMPT_STYLE}"
echo "accept_min_delta:   ${ACCEPT_MIN_DELTA}"
echo "model:              ${MODEL_NAME}"
echo "vllm_max_model_len: ${VLLM_MAX_MODEL_LEN}"
echo "vllm_gpu_mem_util:  ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "vllm_choice_batch:  ${VLLM_CHOICE_BATCH_SIZE}"
echo "output_dir:         ${OUTPUT_DIR}"
echo "========================================="

EM_ARGS=(
  --experiment_dir "${EXP_DIR}"
  --agglomerative_output_dir "${HIERARCHY_DIR}"
  --sentence_data_path "${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"
  --output_file "${OUTPUT_FILE}"
  --diagnostics_dir "${DIAGNOSTICS_DIR}"
  --num_agglomerative_clusters "${N_CLUSTERS}"
  --subsample_seed "${SEED}"
  --model_type vllm
  --model_name "${MODEL_NAME}"
  --sentence_column_name "${SENTENCE_COLUMN_NAME}"
  --num_trials "${NUM_TRIALS}"
  --scorer_type "${SCORER_TYPE}"
  --num_iterations "${N_ITERS}"
  --noop_patience "${NOOP_PATIENCE}"
  --tune_operation "${TUNE_OPERATION}"
  --structural_label_mode "${STRUCTURAL_LABEL_MODE}"
  --label_prompt_style "${LABEL_PROMPT_STYLE}"
  --split_max_per_iter "${SPLIT_MAX_PER_ITER}"
  --split_size_gate_mode "${SPLIT_SIZE_GATE_MODE}"
  --split_min_cluster_size_mean_ratio "${SPLIT_MIN_CLUSTER_SIZE_MEAN_RATIO}"
  --split_gap_top2_max "${SPLIT_GAP_TOP2_MAX}"
  --split_runner_up_min_share "${SPLIT_RUNNER_UP_MIN_SHARE}"
  --merge_max_per_iter "${MERGE_MAX_PER_ITER}"
  --remove_min_cluster_size_mean_ratio "${REMOVE_MIN_CLUSTER_SIZE_MEAN_RATIO}"
  --remove_max_per_iter "${REMOVE_MAX_PER_ITER}"
  --merge_similarity_mode "${MERGE_SIMILARITY_MODE}"
  --merge_similarity_quantile "${MERGE_SIMILARITY_QUANTILE}"
  --merge_epsilon_mode "${MERGE_EPSILON_MODE}"
  --merge_l_abs_diff_quantile "${MERGE_L_ABS_DIFF_QUANTILE}"
  --merge_c_abs_diff_quantile "${MERGE_C_ABS_DIFF_QUANTILE}"
  --add_max_new_clusters_per_iter "${ADD_MAX_NEW_CLUSTERS_PER_ITER}"
  --accept_min_delta "${ACCEPT_MIN_DELTA}"
  --adaptive_threshold_max_relax_iters "${ADAPTIVE_THRESHOLD_MAX_RELAX_ITERS}"
  --adaptive_quantile_step "${ADAPTIVE_QUANTILE_STEP}"
  --adaptive_split_size_relax_ratio "${ADAPTIVE_SPLIT_SIZE_RELAX_RATIO}"
  --adaptive_merge_similarity_relax "${ADAPTIVE_MERGE_SIMILARITY_RELAX}"
  --adaptive_merge_epsilon_expand "${ADAPTIVE_MERGE_EPSILON_EXPAND}"
  --adaptive_remove_size_relax_ratio "${ADAPTIVE_REMOVE_SIZE_RELAX_RATIO}"
  --adaptive_remove_ll_factor_step "${ADAPTIVE_REMOVE_LL_FACTOR_STEP}"
  --proposal_top_k_log "${PROPOSAL_TOP_K_LOG}"
  --log_iteration_metrics
  --log_top_pairs "${LOG_TOP_PAIRS}"
  --diagnostics_sample_size "${DIAGNOSTICS_SAMPLE_SIZE}"
)

if [[ -n "${POINTS_PER_CLUSTER}" ]]; then
  EM_ARGS+=(--num_datapoints_per_cluster "${POINTS_PER_CLUSTER}")
fi
if [[ "${BEST_OPERATION_PER_ITER}" == "1" ]]; then
  EM_ARGS+=(--best_operation_per_iteration)
else
  EM_ARGS+=(--no-best_operation_per_iteration)
fi
if [[ "${SPLIT_ENABLED}" == "1" ]]; then
  EM_ARGS+=(--split_enabled)
else
  EM_ARGS+=(--no-split_enabled)
fi
if [[ "${MERGE_ENABLED}" == "1" ]]; then
  EM_ARGS+=(--merge_enabled)
else
  EM_ARGS+=(--no-merge_enabled)
fi
if [[ "${REMOVE_ENABLED}" == "1" ]]; then
  EM_ARGS+=(--remove_enabled)
else
  EM_ARGS+=(--no-remove_enabled)
fi
if [[ "${ADD_ENABLED}" == "1" ]]; then
  EM_ARGS+=(--add_enabled)
else
  EM_ARGS+=(--no-add_enabled)
fi
if [[ "${REVISE_ENABLED}" == "0" ]]; then
  EM_ARGS+=(--no-revise_enabled)
else
  EM_ARGS+=(--revise_enabled)
fi
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

python src/run_em_algorithm_llm_only.py \
  "${EM_ARGS[@]}"

echo "EM run complete."
echo "Scores CSV: ${OUTPUT_FILE}"
echo "Diagnostics: ${DIAGNOSTICS_DIR}"
