#!/bin/bash
#SBATCH --job-name=em_gt_sanity_llm_only
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/em_gt_sanity_llm_only_%j.out
#SBATCH --error=logs/em_gt_sanity_llm_only_%j.err

###############################################################################
# LLM-only EM on corrupted HuffPost ground-truth schema.
#
# Identical to em_gt_sanity.sh except:
#   1. Calls run_em_algorithm_llm_only.py instead of run_em_algorithm.py
#   2. No sentence-transformer embeddings are used for structural decisions
#      (splits use LLM posterior runner-up disagreement instead of KMeans
#       on ST embeddings; merges use LLM confusion matrix instead of
#       cosine similarity between ST embedding centroids)
#   3. merge_similarity_min is lowered to 0.15 because the LLM-only
#      version measures merge similarity via a confusion matrix
#      (avg symmetric P(z_j|x) between clusters), which is on a very
#      different scale than cosine similarity between ST centroids.
#   4. Uses Llama-3.1-8B-Instruct (tuned for the LLM-only pipeline)
#      instead of Qwen2.5-3B-Instruct → needs 80G RAM.
#   5. Full structural-op configs (split/merge/remove/add/revise)
#      with LLM-only-tuned thresholds from em_llm_only.sh.
#
# Output goes to huffpost_gt_sanity_llm_only/ to keep it separate.
###############################################################################

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"

# ── Experiment config ──────────────────────────────────────────────────────
N_DOCS=1000             # same as random-init run
N_CENTROIDS=200         # same K-means resolution
POINTS_PER_CLUSTER=20   # fewer needed; GT clusters are coherent
N_ITERS=5              # should converge quickly from near-GT start

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# ── Paths ──────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="/nlp/scr/soatto/neural-taxonomy/experiments/huffpost_gt_sanity_llm_only"
# Reuse the same sentence data as the random-init run (same seed, same docs)
SENTENCE_DATA="/nlp/scr/soatto/neural-taxonomy/experiments/huffpost_random_init/huffpost_data.csv"
HIERARCHY_DIR="${EXPERIMENT_DIR}/hierarchy_results"
OUTPUT_FILE="${EXPERIMENT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${EXPERIMENT_DIR}/em_diagnostics"

cd "${REPO_DIR}"
mkdir -p logs "${EXPERIMENT_DIR}" "${DIAGNOSTICS_DIR}"

# ── Cache setup ────────────────────────────────────────────────────────────
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/soatto
else
  LOCAL_SCRATCH=/nlp/scr/soatto
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
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh
conda activate nlp

# ── Memory monitoring (useful for OOM diagnosis with larger model) ─────────
print_mem_snapshot() {
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') memory snapshot ====="
  echo "-- Host memory --"
  free -h || true
  echo "-- GPU memory --"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  echo "=============================================="
}

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

# ── Step 1: Prepare data (only if not already present from prior run) ──────
if [ ! -f "${SENTENCE_DATA}" ]; then
  echo ">>> huffpost_data.csv not found; generating it now..."
  python src/scripts/prepare_huffpost_data.py \
    --output_csv    "${SENTENCE_DATA}" \
    --n_docs        "${N_DOCS}" \
    --n_centroids   "${N_CENTROIDS}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --seed          42
else
  echo ">>> Reusing existing sentence data: ${SENTENCE_DATA}"
fi

# ── Step 2: Build corrupted GT schema ─────────────────────────────────────
echo ">>> Building ground-truth corrupted schema..."
N_CLUSTERS=$(
  python src/scripts/prepare_gt_corrupted_clustering.py \
    --data_csv          "${SENTENCE_DATA}" \
    --output_dir        "${HIERARCHY_DIR}" \
    --top_n_categories  15 \
    --seed              42 \
  | grep '^N_CLUSTERS=' | cut -d= -f2
)
echo "    Detected N_CLUSTERS=${N_CLUSTERS}"

echo "========================================"
echo "HuffPost GT-sanity EM (LLM-only): $(date)"
echo "Model:            ${MODEL_NAME}"
echo "Documents:        ${N_DOCS}"
echo "Centroids:        ${N_CENTROIDS}"
echo "Starting clusters: ${N_CLUSTERS}  (GT top-15 cats with 2 merges / 2 splits / 2 mislabels)"
echo "Points/cluster:   ${POINTS_PER_CLUSTER}  (total: $((N_CLUSTERS * POINTS_PER_CLUSTER)))"
echo "EM iterations:    ${N_ITERS}"
echo "Algorithm:        LLM-only (no ST embeddings for structural decisions)"
echo "========================================"

# ── Step 3: Run EM (LLM-only) ─────────────────────────────────────────────
# Same flags as em_gt_sanity.sh, just calling run_em_algorithm_llm_only.py.
# The one threshold that legitimately differs (merge_similarity_min 0.15 vs
# 0.80) is already the default in run_em_algorithm_llm_only.py.
echo ">>> Running LLM-only EM..."
python src/run_em_algorithm_llm_only.py \
  --experiment_dir             "${EXPERIMENT_DIR}" \
  --sentence_data_path         "${SENTENCE_DATA}" \
  --output_file                "${OUTPUT_FILE}" \
  --diagnostics_dir            "${DIAGNOSTICS_DIR}" \
  --num_agglomerative_clusters "${N_CLUSTERS}" \
  --model_type                 vllm \
  --model_name                 "${MODEL_NAME}" \
  --sentence_column_name       description \
  --num_trials                 1 \
  --scorer_type                batch \
  --num_iterations             "${N_ITERS}" \
  --noop_patience              10 \
  --split_max_per_iter         3 \
  --split_cooldown_iters       2 \
  --num_datapoints_per_cluster "${POINTS_PER_CLUSTER}" \
  --baseline_sample_size       200 \
  --log_iteration_metrics \
  --show_progress \
  --embedding_model_name       "${EMBEDDING_MODEL}" \
  --log_top_pairs              5 \
  --diagnostics_sample_size    200 \
  --diagnostics_bins           40

echo "========================================"
echo "Done: $(date)"
echo "========================================"

print_mem_snapshot
