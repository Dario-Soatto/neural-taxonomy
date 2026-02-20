#!/bin/bash
#SBATCH --job-name=em_wiki_fast
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=4:00:00
#SBATCH --output=logs/em_fast_%j.out
#SBATCH --error=logs/em_fast_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"
EXPERIMENT_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy/experiments/wiki_biographies_10000"
OUTPUT_FILE="/nlp/scr/soatto/neural-taxonomy/em_refined_scores_fast.csv"
DIAGNOSTICS_DIR="/nlp/scr/soatto/neural-taxonomy/em_diagnostics"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

# Avoid /sailhome quota by putting caches on local scratch
export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"

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

# Ensure max vLLM seq length is honored
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh  # Update to your conda path
conda activate nlp

echo "========================================"
echo "Starting FAST EM run at $(date)"
echo "Model: ${MODEL_NAME}"
echo "Experiment: ${EXPERIMENT_DIR}"
echo "========================================"

mkdir -p "${DIAGNOSTICS_DIR}"

python src/run_em_algorithm.py \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --diagnostics_dir "${DIAGNOSTICS_DIR}" \
  --num_agglomerative_clusters 10 \
  --model_type vllm \
  --model_name "${MODEL_NAME}" \
  --sentence_column_name description \
  --num_trials 1 \
  --scorer_type batch \
  --num_iterations 5 \
  --num_datapoints_per_cluster 50 \
  --baseline_sample_size 200 \
  --max_texts_per_cluster_metrics -1 \
  --log_iteration_metrics \
  --show_progress \
  --embedding_model_name sentence-transformers/all-MiniLM-L6-v2 \
  --log_top_pairs 10 \
  --diagnostics_sample_size 200 \
  --diagnostics_bins 40

echo "========================================"
echo "Finished at $(date)"
echo "========================================"