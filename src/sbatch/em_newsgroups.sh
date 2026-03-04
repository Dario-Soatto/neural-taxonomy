#!/bin/bash
#SBATCH --job-name=em_newsgroups
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00
#SBATCH --output=logs/em_newsgroups_%j.out
#SBATCH --error=logs/em_newsgroups_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"

# ── Experiment config ──────────────────────────────────────────────────────
N_DOCS=1000             # number of HuffPost headlines to use
N_CENTROIDS=200         # K-means centroids (fine-grained micro-clusters)
N_CLUSTERS=15           # random starting topics (HuffPost has 41 true categories)
POINTS_PER_CLUSTER=50   # docs sampled per cluster per EM iteration
N_ITERS=20

MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# ── Paths ──────────────────────────────────────────────────────────────────
EXPERIMENT_DIR="/nlp/scr/soatto/neural-taxonomy/experiments/huffpost_random_init"
SENTENCE_DATA="${EXPERIMENT_DIR}/huffpost_data.csv"
HIERARCHY_DIR="${EXPERIMENT_DIR}/hierarchy_results"
OUTPUT_FILE="${EXPERIMENT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${EXPERIMENT_DIR}/em_diagnostics"

cd "${REPO_DIR}"
mkdir -p logs "${EXPERIMENT_DIR}" "${DIAGNOSTICS_DIR}"

# ── Cache setup ────────────────────────────────────────────────────────────
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
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "========================================"
echo "HuffPost News EM (random init): $(date)"
echo "Model:            ${MODEL_NAME}"
echo "Documents:        ${N_DOCS}"
echo "Centroids:        ${N_CENTROIDS}"
echo "Starting clusters: ${N_CLUSTERS}"
echo "Points/cluster:   ${POINTS_PER_CLUSTER}  (total: $((N_CLUSTERS * POINTS_PER_CLUSTER)))"
echo "EM iterations:    ${N_ITERS}"
echo "========================================"

# Step 1: Prepare HuffPost data (embed + K-means → CSV)
echo ">>> Preparing HuffPost News data..."
python src/scripts/prepare_huffpost_data.py \
  --output_csv    "${SENTENCE_DATA}" \
  --n_docs        "${N_DOCS}" \
  --n_centroids   "${N_CENTROIDS}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --seed          42

# Step 2: Generate random starting schema
echo ">>> Generating random cluster assignments..."
python src/scripts/prepare_random_clustering.py \
  --n_centroids   "${N_CENTROIDS}" \
  --n_clusters    "${N_CLUSTERS}" \
  --output_dir    "${HIERARCHY_DIR}" \
  --seed          42

# Step 3: Run EM
echo ">>> Running EM..."
python src/run_em_algorithm.py \
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
  --noop_patience              20 \
  --split_max_per_iter         3 \
  --split_cooldown_iters       3 \
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
