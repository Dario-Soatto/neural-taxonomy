#!/bin/bash
#SBATCH --job-name=em_thresholds
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --nodelist=jagupard29
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/thresholds_%j.out
#SBATCH --error=logs/thresholds_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
EXPERIMENT_DIR="experiments/wiki_biographies_1000"
CLUSTERS_FILE="experiments/wiki_biographies_1000/hierarchy_results__standard/clusters_level_3_4_clusters.csv"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/tdalmia
else
  LOCAL_SCRATCH=/nlp/scr/tdalmia
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

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

python scripts/thresholds.py \
  --experiment_dir "${EXPERIMENT_DIR}" \
  --clusters_file "${CLUSTERS_FILE}" \
  --text_col description \
  --embedding_model sentence-transformers/all-MiniLM-L6-v2
