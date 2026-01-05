#!/bin/bash
#SBATCH --job-name=nt_smoke
#SBATCH --partition=john
#SBATCH --account=nlp
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nlp/scr/tdalmia/projects/neural-taxonomy/logs/nt_smoke_%j.out
#SBATCH --error=/nlp/scr/tdalmia/projects/neural-taxonomy/logs/nt_smoke_%j.err

set -euo pipefail

cd /nlp/scr/tdalmia/projects/neural-taxonomy

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

export HF_HOME=/nlp/scr/tdalmia/projects/neural-taxonomy/.cache/hf
export TRANSFORMERS_CACHE=/nlp/scr/tdalmia/projects/neural-taxonomy/.cache/transformers
export HF_DATASETS_CACHE=/nlp/scr/tdalmia/projects/neural-taxonomy/.cache/datasets
export TOKENIZERS_PARALLELISM=false

: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set}"

EXP_NAME=editorial
EXP_DIR=experiments/${EXP_NAME}
MODEL_NAME="gpt-4o-mini"

mkdir -p "${EXP_DIR}" "${EXP_DIR}/data" "${EXP_DIR}/models" logs

INPUT_FILE="${EXP_DIR}/editorial-discourse-input-data.csv"
STEP1_OUT="${EXP_DIR}/custom-initial-labeling.json"

ln -sf /nlp/scr/tdalmia/projects/neural-taxonomy/processed_custom_input.csv "${INPUT_FILE}"

SIM_OUT="${EXP_DIR}/${MODEL_NAME//\//-}-similarity-data.jsonl"
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
SBERT_OUT="${EXP_DIR}/models/custom-sentence-similarity-model"

python src/step_1__run_initial_labeling_prompts.py \
  --model "${MODEL_NAME}" \
  --input_data_file "${INPUT_FILE}" \
  --output_file "${STEP1_OUT}" \
  --experiment "${EXP_NAME}" \
  --start_idx 0 \
  --end_idx 10 \
  --batch_size 10 \
  --num_sents_per_prompt 2 \
  --use_openai

STEP1_LABELS="${EXP_DIR}/custom-initial-labeling-labeling__experiment-${EXP_NAME}__model_${MODEL_NAME//\//-}__0_10.json"

python src/step_2__create_supervised_similarity_data.py \
  --input_file "${STEP1_LABELS}" \
  --output_file "${SIM_OUT}" \
  --model_name "${MODEL_NAME}" \
  --use_openai \
  --batch_size 50 \
  --text_col_name label \
  --text_col_name_2 description \
  --sample_size 200 \
  --k 3 \
  --min_sim_threshold 0.0 \
  --max_sim_threshold 1.0

TRIPLETS_FILE="${EXP_DIR}/triplets_${MODEL_NAME//\//-}-similarity-data.jsonl"

python src/step_3__train_sentence_similarity_model.py \
  --model_name "${SBERT_MODEL}" \
  --data_file "${TRIPLETS_FILE}" \
  --output_dir "${SBERT_OUT}" \
  --num_train_epochs 1 \
  --train_batch_size 4 \
  --eval_batch_size 4 \
  --eval_strategy "epoch" \
  --save_strategy "epoch" \
  --train_subset_size 200 \
  --debug

python src/step_4__merge_labels.py \
  --input_data_file "${STEP1_LABELS}" \
  --input_col_name label \
  --trained_sbert_model_name "${SBERT_OUT}/trained-model" \
  --output_cluster_file "${EXP_DIR}/models/cluster_centroids.npy" \
  --output_data_file "${EXP_DIR}/models/all_extracted_discourse_with_clusters.csv" \
  --skip_umap \
  --ncentroids 8 \
  --niter 20 \
  --kmeans_downsample_to 100

python src/step_5__label_low_level_kmeans_clusters.py \
  --input_file "${EXP_DIR}/models/all_extracted_discourse_with_clusters.csv" \
  --output_file "${EXP_DIR}/models/cluster_labels.csv" \
  --model "${MODEL_NAME}" \
  --cluster_col cluster \
  --label_superset_col label \
  --n_samples_per_cluster 3 \
  --use_openai

mkdir -p "${EXP_DIR}/models/agglomerative_clustering_outputs"

python src/step_6__agglomerative_clustering.py \
  "${EXP_DIR}/models/cluster_centroids.npy" \
  "${EXP_DIR}/models/agglomerative_clustering_outputs" \
  --min_clusters 2 \
  --max_clusters 4 \
  --min_cluster_size 2 \
  --method ward \
  --metric euclidean

echo "Smoke test completed."
