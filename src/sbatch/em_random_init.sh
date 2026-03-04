#!/bin/bash
#SBATCH --job-name=em_random_init
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/em_random_init_%j.out
#SBATCH --error=logs/em_random_init_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"

# Tushar's sentence data (unchanged — same text as em_fast uses)
SENTENCE_DATA="/nlp/scr/tdalmia/projects/neural-taxonomy/experiments/wiki_biographies_10000/models/all_extracted_discourse_with_clusters.csv"
N_CENTROIDS=1024   # number of K-means centroids in that CSV; adjust if wrong

# New output locations (separate from Tushar's dir)
EXPERIMENT_DIR="/nlp/scr/soatto/neural-taxonomy/experiments/wiki_random_init"
HIERARCHY_DIR="${EXPERIMENT_DIR}/hierarchy_results"
OUTPUT_FILE="${EXPERIMENT_DIR}/em_refined_scores.csv"
DIAGNOSTICS_DIR="${EXPERIMENT_DIR}/em_diagnostics"
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

N_CLUSTERS=10
POINTS_PER_CLUSTER=50   # 50 × 10 = 500 total  (same speed as em_fast)
N_ITERS=20              # run up to 50 iterations

cd "${REPO_DIR}"
mkdir -p logs "${EXPERIMENT_DIR}" "${DIAGNOSTICS_DIR}"

# ── Cache setup (identical to em_fast.sh) ─────────────────────────────────
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
echo "RANDOM-INIT EM sanity check: $(date)"
echo "Model:            ${MODEL_NAME}"
echo "Clusters:         ${N_CLUSTERS}"
echo "Points/cluster:   ${POINTS_PER_CLUSTER}  (total: $((N_CLUSTERS * POINTS_PER_CLUSTER)))"
echo "EM iterations:    ${N_ITERS}"
echo "========================================"

# Step 1: Generate random schema files
echo ">>> Generating random cluster assignments..."
python src/scripts/prepare_random_clustering.py \
  --n_centroids "${N_CENTROIDS}" \
  --n_clusters  "${N_CLUSTERS}" \
  --output_dir  "${HIERARCHY_DIR}" \
  --seed        42

# Step 2: Run EM from random start
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
  --embedding_model_name       sentence-transformers/all-MiniLM-L6-v2 \
  --log_top_pairs              5 \
  --diagnostics_sample_size    200 \
  --diagnostics_bins           40

echo "========================================"
echo "Done: $(date)"
echo "========================================"