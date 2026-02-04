#!/bin/bash
#SBATCH --job-name=step4_kmeans_wiki
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard29
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step4_kmeans_%j.out
#SBATCH --error=logs/step4_kmeans_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
INPUT_GLOB="experiments/wiki_biographies_10000/wiki_biographies-initial-labeling-labeling__experiment-biographies__model_meta-llama-Meta-Llama-3.1-8B-Instruct__*.json"
MERGED_INPUT="experiments/wiki_biographies_10000/wiki_biographies-initial-labeling-labeling__experiment-biographies__model_meta-llama-Meta-Llama-3.1-8B-Instruct__merged.jsonl"
TRAINED_SBERT="experiments/wiki_biographies_10000/models/wiki-sentence-similarity-model/trained-model"
OUTPUT_CLUSTER_FILE="experiments/wiki_biographies_10000/models/cluster_centroids.npy"
OUTPUT_DATA_FILE="experiments/wiki_biographies_10000/models/all_extracted_discourse_with_clusters.csv"

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

python3 - <<'PY'
import glob
import os

input_glob = "experiments/wiki_biographies_10000/wiki_biographies-initial-labeling-labeling__experiment-biographies__model_meta-llama-Meta-Llama-3.1-8B-Instruct__*.json"
merged_path = "experiments/wiki_biographies_10000/wiki_biographies-initial-labeling-labeling__experiment-biographies__model_meta-llama-Meta-Llama-3.1-8B-Instruct__merged.jsonl"

files = sorted(glob.glob(input_glob))
if not files:
    raise SystemExit(f"No shard files found for pattern: {input_glob}")

os.makedirs(os.path.dirname(merged_path), exist_ok=True)
with open(merged_path, "w") as out_f:
    for path in files:
        with open(path, "r") as in_f:
            for line in in_f:
                line = line.strip()
                if line:
                    out_f.write(line + "\n")
print(f"Merged {len(files)} files into {merged_path}")
PY

python src/step_4__merge_labels.py \
  --input_data_file "${MERGED_INPUT}" \
  --input_col_name label \
  --trained_sbert_model_name "${TRAINED_SBERT}" \
  --output_cluster_file "${OUTPUT_CLUSTER_FILE}" \
  --output_data_file "${OUTPUT_DATA_FILE}" \
  --skip_umap \
  --ncentroids 100 \
  --niter 50 \
  --sbert_batch_size 64 \
  # --n_rows_to_process 10000
