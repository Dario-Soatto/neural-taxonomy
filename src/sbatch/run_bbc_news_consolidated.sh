#!/bin/bash
#SBATCH --job-name=bbc_news_all
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --gres=gpu:a6000:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=16:00:00
#SBATCH --output=bbc_news_consolidated_%j.out
#SBATCH --error=bbc_news_consolidated_%j.err
#
# BBC News: full pipeline (steps 0–7) in one job, with fixes from cluster debugging:
#   - REPO_DIR = SLURM_SUBMIT_DIR (always run: cd /path/to/neural-taxonomy && sbatch …)
#   - A6000 GPU (avoid 12GB cards that OOM on Llama 8B)
#   - HF + vLLM caches on scratch (avoid /sailhome disk quota on Hub downloads / usage stats)
#   - Preflight: numpy/faiss ABI, accelerate, more-itertools, datasets
#   - Skip completed steps when outputs already exist (safe resume)
#   - Logs in repo root (no logs/ dir required before sbatch)
#
# Usage:
#   cd /path/to/neural-taxonomy && sbatch src/sbatch/run_bbc_news_consolidated.sh
#
# Override conda root if needed:
#   export CONDA_ROOT=/path/to/miniconda3 && sbatch src/sbatch/run_bbc_news_consolidated.sh
#

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi

cd "${REPO_DIR}"
echo "========================================"
echo "REPO_DIR=${REPO_DIR}"
echo "GIT_HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo 'not a git repo')"
echo "START $(date)"
echo "========================================"

EXPERIMENT_NAME="bbc_news"
EXPERIMENT_DIR="${REPO_DIR}/experiments/${EXPERIMENT_NAME}"
MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
SBERT_MODEL="sentence-transformers/all-MiniLM-L6-v2"
NUM_CENTROIDS=64
STEP1_BATCH_SIZE=2500
STEP2_VLLM_BATCH=5000
STEP6_HIER_SUFFIX="hierarchy_results__standard"

mkdir -p "${EXPERIMENT_DIR}/data" "${EXPERIMENT_DIR}/models"

# ── Scratch for caches (prefer large local disks; never rely on sailhome quota) ──
if [[ -d "/scr-ssd/${USER}" ]]; then
  LOCAL_SCRATCH="/scr-ssd/${USER}"
elif [[ -d "/nlp/scr/${USER}" ]]; then
  LOCAL_SCRATCH="/nlp/scr/${USER}"
elif [[ -d "/juice2/scr2/${USER}" ]]; then
  LOCAL_SCRATCH="/juice2/scr2/${USER}"
else
  LOCAL_SCRATCH="${SLURM_TMPDIR:-/tmp}/nt-cache-${USER}"
fi
mkdir -p "${LOCAL_SCRATCH}/hf_cache/hub" "${LOCAL_SCRATCH}/cache/vllm"

export HF_HOME="${LOCAL_SCRATCH}/hf_cache"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="${LOCAL_SCRATCH}/cache"
export TORCH_HOME="${LOCAL_SCRATCH}/cache/torch"
export TORCH_EXTENSIONS_DIR="${LOCAL_SCRATCH}/cache/torch_extensions"
export PIP_CACHE_DIR="${LOCAL_SCRATCH}/cache/pip"

export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export VLLM_NO_USAGE_STATS=1
export VLLM_USAGE_STATS_PATH="${LOCAL_SCRATCH}/cache/vllm/usage_stats.json"

export VLLM_MAX_MODEL_LEN=4096

if [[ -z "${HF_TOKEN:-}" ]]; then
  for _tok_path in "${HOME}/.cache/huggingface/token" "${HOME}/.huggingface/token"; do
    if [[ -f "${_tok_path}" ]]; then
      export HF_TOKEN="$(cat "${_tok_path}")"
      break
    fi
  done
fi

# ── Conda ───────────────────────────────────────────────────────────────────
if [[ -n "${CONDA_ROOT:-}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/nlp/scr/${USER}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/nlp/scr/${USER}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "/nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/nlp/scr/soatto/miniconda3/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found. Set CONDA_ROOT to your Miniconda/Anaconda root." >&2
  exit 1
fi
conda activate nlp

# ── Python deps (small wheels; needs outbound network on the node) ─────────
pip install -q 'accelerate>=0.26.0' 'more-itertools' 'datasets' || {
  echo "WARN: pip install failed; ensure accelerate, more-itertools, datasets are in env." >&2
}

# ── faiss / numpy ABI (NumPy 2 + old faiss wheels → import errors) ─────────
python <<'PYCHECK'
import sys
try:
    import numpy as np
    import faiss  # noqa: F401
except Exception as e:
    print("ERROR: faiss import failed:", e, file=sys.stderr)
    print("Fix on the login node, then resubmit:", file=sys.stderr)
    print("  pip install 'numpy>=1.26.4,<2' --force-reinstall", file=sys.stderr)
    print("  pip uninstall -y faiss faiss-cpu faiss-gpu 2>/dev/null; pip install --no-cache-dir faiss-cpu", file=sys.stderr)
    sys.exit(1)
maj = int(__import__("numpy").__version__.split(".")[0])
if maj >= 2:
    print("ERROR: NumPy 2.x often breaks faiss. Pin NumPy < 2 (see messages above).", file=sys.stderr)
    sys.exit(1)
PYCHECK

# ── Paths used in all steps ─────────────────────────────────────────────────
INPUT_CSV="${EXPERIMENT_DIR}/data/bbc_news.csv"
TRIPLETS_JSONL="${EXPERIMENT_DIR}/triplets_vllm-similarity-data.jsonl"
SBERT_OUT="${EXPERIMENT_DIR}/models/${EXPERIMENT_NAME}-sentence-similarity-model"
SBERT_TRAINED="${SBERT_OUT}/trained-model"
STEP4_CSV="${EXPERIMENT_DIR}/models/all_extracted_discourse_with_clusters.csv"
CLUSTER_LABELS="${EXPERIMENT_DIR}/models/cluster_labels.csv"
EVAL_CSV="${EXPERIMENT_DIR}/evaluation_results.csv"

# ── Step 0 ──────────────────────────────────────────────────────────────────
if [[ -f "${INPUT_CSV}" ]]; then
  echo ">>> Step 0: SKIP (found ${INPUT_CSV})"
else
  echo ">>> Step 0: Preparing BBC News CSV…"
  python src/scripts/prepare_bbc_news_data.py --output_csv "${INPUT_CSV}"
fi

# ── Step 1 ─────────────────────────────────────────────────────────────────
shopt -s nullglob
STEP1_CAND=( "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling-labeling__experiment-bbc-news__model_${MODEL_NAME//\//-}__"*.json )
shopt -u nullglob
STEP1_RESULT=""
if ((${#STEP1_CAND[@]})) && [[ -s "${STEP1_CAND[0]}" ]]; then
  STEP1_RESULT="${STEP1_CAND[0]}"
fi
if [[ -n "${STEP1_RESULT}" ]]; then
  echo ">>> Step 1: SKIP (found ${STEP1_RESULT})"
else
  echo ">>> Step 1: Initial labeling (vLLM)…"
  python src/step_1__run_initial_labeling_prompts.py \
    --model "${MODEL_NAME}" \
    --input_data_file "${INPUT_CSV}" \
    --output_file "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling.json" \
    --experiment bbc-news \
    --batch_size "${STEP1_BATCH_SIZE}" \
    --num_sents_per_prompt 1
  shopt -s nullglob
  STEP1_CAND=( "${EXPERIMENT_DIR}/${EXPERIMENT_NAME}-initial-labeling-labeling__experiment-bbc-news__model_${MODEL_NAME//\//-}__"*.json )
  shopt -u nullglob
  STEP1_RESULT="${STEP1_CAND[0]:-}"
fi
if [[ -z "${STEP1_RESULT}" || ! -s "${STEP1_RESULT}" ]]; then
  echo "ERROR: Step 1 produced no usable JSON shard." >&2
  exit 1
fi
echo "Step 1 shard: ${STEP1_RESULT}"

# ── Step 2 ─────────────────────────────────────────────────────────────────
if [[ -s "${TRIPLETS_JSONL}" ]]; then
  echo ">>> Step 2: SKIP (found ${TRIPLETS_JSONL})"
else
  echo ">>> Step 2: Similarity prompts + vLLM + triplets…"
  python src/step_2__create_supervised_similarity_data.py \
    --input_file "${STEP1_RESULT}" \
    --output_file "${EXPERIMENT_DIR}/vllm-similarity-data.jsonl" \
    --model_name "${MODEL_NAME}" \
    --batch_size "${STEP2_VLLM_BATCH}" \
    --prompt_template generic \
    --text_col_name label \
    --text_col_name_2 description \
    --sample_size 500000
fi

# ── Step 3 ─────────────────────────────────────────────────────────────────
if [[ -f "${SBERT_TRAINED}/model.safetensors" ]]; then
  echo ">>> Step 3: SKIP (found ${SBERT_TRAINED}/model.safetensors)"
else
  echo ">>> Step 3: Fine-tune SBERT…"
  python src/step_3__train_sentence_similarity_model.py \
    --model_name "${SBERT_MODEL}" \
    --data_file "${TRIPLETS_JSONL}" \
    --output_dir "${SBERT_OUT}" \
    --num_train_epochs 3 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --eval_strategy epoch \
    --save_strategy epoch \
    --run_name "${EXPERIMENT_NAME}-sentence-similarity-model" \
    --do_initial_evaluation
fi

# ── Step 4 ─────────────────────────────────────────────────────────────────
if [[ -f "${STEP4_CSV}" ]]; then
  echo ">>> Step 4: SKIP (found ${STEP4_CSV})"
else
  echo ">>> Step 4: Embed + K-means…"
  python src/step_4__merge_labels.py \
    --input_data_file "${STEP1_RESULT}" \
    --input_col_name label \
    --trained_sbert_model_name "${SBERT_TRAINED}" \
    --output_cluster_file "${EXPERIMENT_DIR}/models/cluster_centroids.npy" \
    --output_data_file "${STEP4_CSV}" \
    --skip_umap \
    --ncentroids "${NUM_CENTROIDS}"
fi

# ── Step 5 ─────────────────────────────────────────────────────────────────
if [[ -f "${CLUSTER_LABELS}" ]]; then
  echo ">>> Step 5: SKIP (found ${CLUSTER_LABELS})"
else
  echo ">>> Step 5: LLM labels for K-means clusters…"
  python src/step_5__label_low_level_kmeans_clusters.py \
    --input_file "${STEP4_CSV}" \
    --output_file "${CLUSTER_LABELS}" \
    --model "${MODEL_NAME}" \
    --cluster_col cluster \
    --label_superset_col label \
    --n_samples_per_cluster 10
fi

# ── Step 6 (writes ${EXPERIMENT_DIR}/hierarchy_results__standard/ by default) ─
STEP6_THRESH="${EXPERIMENT_DIR}/${STEP6_HIER_SUFFIX}/optimal_thresholds.csv"
if [[ -f "${STEP6_THRESH}" ]]; then
  echo ">>> Step 6: SKIP (found ${STEP6_THRESH})"
else
  echo ">>> Step 6: Agglomerative clustering…"
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
    --initial_cluster_labels_path "${CLUSTER_LABELS}" \
    --no_visualize
fi

# ── Step 7 ─────────────────────────────────────────────────────────────────
if [[ -f "${EVAL_CSV}" ]]; then
  echo ">>> Step 7: SKIP (found ${EVAL_CSV})"
else
  echo ">>> Step 7: NMI / ARI / purity vs true_label…"
  python src/scripts/evaluate_pipeline.py \
    --experiment_dir "${EXPERIMENT_DIR}" \
    --raw_data_csv "${INPUT_CSV}" \
    --true_label_col true_label
fi

echo "========================================"
echo "DONE $(date)"
echo "Metrics: ${EVAL_CSV}"
echo "========================================"
