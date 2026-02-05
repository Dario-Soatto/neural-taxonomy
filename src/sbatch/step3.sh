#!/bin/bash
#SBATCH --job-name=step3_sbert_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step3_sbert_%j.out
#SBATCH --error=logs/step3_sbert_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
DATA_FILE="experiments/wiki_biographies_10000/triplets_vllm-similarity-data.jsonl"
OUTPUT_DIR="experiments/wiki_biographies_10000/models/wiki-sentence-similarity-model"
MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

cd "${REPO_DIR}"
mkdir -p logs

# Prefer node-local SSD if available to reduce load on shared filesystems
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/tdalmia
else
  LOCAL_SCRATCH=/nlp/scr/tdalmia
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/torch_extensions,cache/pip}

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

python src/step_3__train_sentence_similarity_model.py \
  --model_name "${MODEL_NAME}" \
  --data_file "${DATA_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 1 \
  --train_batch_size 16 \
  --eval_batch_size 16 \
  # --train_subset_size 50000 \
  --eval_strategy steps \
  --eval_steps 200 \
  --save_strategy steps \
  --save_steps 200 \
  --run_name wiki-sentence-similarity-model
