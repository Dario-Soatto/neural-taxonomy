#!/bin/bash
#SBATCH --job-name=bbc_news_pipeline
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --output=logs/bbc_news_%j.out
#SBATCH --error=logs/bbc_news_%j.err

###############################################################################
# Full 6-step pipeline for BBC News (2,225 docs, 5 categories).
#
# Usage:
#   sbatch src/sbatch/run_bbc_news.sh
###############################################################################

set -euo pipefail

REPO_DIR="/nlp/scr/soatto/neural-taxonomy"
EXPERIMENT_NAME="bbc_news"
EXPERIMENT_DIR="${REPO_DIR}/experiments/${EXPERIMENT_NAME}"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
NUM_CENTROIDS=64
BATCH_SIZE=2500

cd "${REPO_DIR}"
mkdir -p logs "${EXPERIMENT_DIR}/data" "${EXPERIMENT_DIR}/models"

# ── Cache setup (keep caches off /sailhome) ─────────────────────────────────
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/flashinfer,cache/torch,cache/torch_extensions,cache/pip,cache/vllm}

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export HF_DATASETS_CACHE="$LOCAL_SCRATCH/hf_cache/datasets"
export XDG_CACHE_HOME="$LOCAL_SCRATCH/cache"
export FLASHINFER_WORKSPACE_DIR="$LOCAL_SCRATCH/cache/flashinfer"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/cache/torch_extensions"
export PIP_CACHE_DIR="$LOCAL_SCRATCH/cache/pip"
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="$LOCAL_SCRATCH/cache/vllm/usage_stats.json"

if [ -z "${HF_TOKEN:-}" ]; then
    for _tok_path in "$HOME/.cache/huggingface/token" "$HOME/.huggingface/token"; do
        if [ -f "$_tok_path" ]; then
            export HF_TOKEN="$(cat "$_tok_path")"
            break
        fi
    done
fi
export HOME="$LOCAL_SCRATCH"
export VLLM_MAX_MODEL_LEN=4096

source /nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh
conda activate nlp

echo "========================================"
echo "BBC News 6-step pipeline: $(date)"
echo "Model:      ${MODEL_NAME}"
echo "Centroids:  ${NUM_CENTROIDS}"
echo "========================================"

# ── Step 0: Prepare data ────────────────────────────────────────────────────
INPUT_CSV="${EXPERIMENT_DIR}/data/bbc_news.csv"
if [ ! -f "${INPUT_CSV}" ]; then
    echo ">>> Step 0: Downloading BBC News dataset..."
    python src/scripts/prepare_bbc_news_data.py \
        --output_csv "${INPUT_CSV}"
else
    echo ">>> Step 0: Data already exists at ${INPUT_CSV}, skipping."
fi

# ── Step 1: Initial labeling ────────────────────────────────────────────────
STEP1_OUTPUT="${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling.json"
echo ">>> Step 1: Running initial labeling..."
python src/step_1__run_initial_labeling_prompts.py \
    --model "${MODEL_NAME}" \
    --input_data_file "${INPUT_CSV}" \
    --output_file "${STEP1_OUTPUT}" \
    --experiment bbc-news \
    --batch_size ${BATCH_SIZE} \
    --num_sents_per_prompt 1 \
    --overwrite_prompt_cache

# With BATCH_SIZE >= dataset size, step 1 produces a single shard.
STEP1_RESULT=$(ls ${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling-labeling__experiment-bbc-news__model_${MODEL_NAME//\//-}__*.json 2>/dev/null | head -n 1)
if [ -z "${STEP1_RESULT}" ]; then
    echo "ERROR: No step 1 output found. Check logs." >&2
    exit 1
fi
echo "Step 1 output: ${STEP1_RESULT}"

# ── Step 2: Supervised similarity data ──────────────────────────────────────
echo ">>> Step 2: Creating supervised similarity data..."
python src/step_2__create_supervised_similarity_data.py \
    --input_file "${STEP1_RESULT}" \
    --output_file "${EXPERIMENT_DIR}/vllm-similarity-data.jsonl" \
    --model_name "${MODEL_NAME}" \
    --batch_size 5000 \
    --prompt_template generic \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 500000

# ── Step 3: Train SBERT ────────────────────────────────────────────────────
echo ">>> Step 3: Training sentence similarity model..."
python src/step_3__train_sentence_similarity_model.py \
    --model_name "${SBERT_MODEL}" \
    --data_file "${EXPERIMENT_DIR}/triplets_vllm-similarity-data.jsonl" \
    --output_dir "${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}-sentence-similarity-model" \
    --num_train_epochs 3 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --run_name "${EXPERIMENT_NAME}-sentence-similarity-model" \
    --do_initial_evaluation

# ── Step 4: Embed + K-means ────────────────────────────────────────────────
echo ">>> Step 4: Running clustering..."
python src/step_4__merge_labels.py \
    --input_data_file "${STEP1_RESULT}" \
    --input_col_name label \
    --trained_sbert_model_name "${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}-sentence-similarity-model/trained-model" \
    --output_cluster_file "${EXPERIMENT_DIR}/models/cluster_centroids.npy" \
    --output_data_file "${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters.csv" \
    --skip_umap \
    --ncentroids ${NUM_CENTROIDS}

# ── Step 5: Label K-means clusters ─────────────────────────────────────────
echo ">>> Step 5: Labeling clusters..."
python src/step_5__label_low_level_kmeans_clusters.py \
    --input_file "${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters.csv" \
    --output_file "${EXPERIMENT_DIR}/models/cluster_labels.csv" \
    --model "${MODEL_NAME}" \
    --cluster_col cluster \
    --label_superset_col label \
    --n_samples_per_cluster 10

# ── Step 6: Agglomerative clustering ───────────────────────────────────────
echo ">>> Step 6: Running agglomerative clustering..."
mkdir -p "${EXPERIMENT_DIR}/hierarchy_results"
python src/step_6__agglomerative_clustering.py \
    "${EXPERIMENT_DIR}/models/cluster_centroids.npy" \
    "${EXPERIMENT_DIR}/hierarchy_results" \
    --min_clusters 2 \
    --max_clusters 15 \
    --min_cluster_size 2 \
    --method ward \
    --metric euclidean \
    --experiment bbc-news \
    --initial_cluster_labels_path "${EXPERIMENT_DIR}/models/cluster_labels.csv" \
    --no_visualize

# ── Step 7: Evaluate against ground truth ───────────────────────────────────
echo ">>> Step 7: Evaluating against ground truth..."
python src/scripts/evaluate_pipeline.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --raw_data_csv "${INPUT_CSV}" \
    --true_label_col true_label

echo "========================================"
echo "BBC News pipeline complete: $(date)"
echo "========================================"
