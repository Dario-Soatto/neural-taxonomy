#!/bin/bash
# End-to-end baseline benchmark for newsgroups.
# Runs: data prep -> baselines (CPU) -> LLM scoring (GPU) -> collation.
#
# Note: This requires GPU. If running on a CPU-only node, run
# baselines_cpu.sh first, then baselines_llm_score.sh on a GPU node.
#
# Usage:
#   sbatch src/sbatch/baselines_full_newsgroups.sh

#SBATCH --job-name=baselines_full_ng
#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --nodelist=jagupard[28-39]
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baselines_full_ng_%j.out
#SBATCH --error=logs/baselines_full_ng_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/nlp/scr/tdalmia/projects/neural-taxonomy}"
EXP_DIR="experiments/newsgroups_sanity_tiny"
MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
NUM_TRIALS="${NUM_TRIALS:-3}"
N_RUNS="${N_RUNS:-3}"
METHODS="${METHODS:-lda_gibbs lda_vi embedding_kmeans bertopic}"

cd "${REPO_DIR}"
mkdir -p logs results

# Setup caches
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi

mkdir -p "$LOCAL_SCRATCH"/{hf_cache,cache,cache/torch,cache/vllm}

export HF_HOME="$LOCAL_SCRATCH/hf_cache"
export HUGGINGFACE_HUB_CACHE="$LOCAL_SCRATCH/hf_cache/hub"
export TRANSFORMERS_CACHE="$LOCAL_SCRATCH/hf_cache/transformers"
export TORCH_HOME="$LOCAL_SCRATCH/cache/torch"
export VLLM_USAGE_STATS=0
export VLLM_DISABLE_USAGE_STATS=1
export HOME="$LOCAL_SCRATCH"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

echo "========================================"
echo "Full newsgroups baseline benchmark"
echo "Started:     $(date)"
echo "Methods:     ${METHODS}"
echo "Model:       ${MODEL_NAME}"
echo "========================================"

# Step 1: Prepare data (if needed)
DATA_CSV="${EXP_DIR}/models/all_extracted_discourse_with_clusters_and_text.csv"
if [[ ! -f "${DATA_CSV}" ]]; then
  echo "Step 1: Preparing newsgroups data..."
  mkdir -p "${EXP_DIR}/models"

  python src/scripts/prepare_newsgroups_data.py \
    --output_csv "${DATA_CSV}" \
    --n_docs 240 \
    --n_centroids 60 \
    --max_words 1200 \
    --subset all \
    --seed 13

  python src/scripts/prepare_newsgroups_gt_hierarchies.py \
    --data_csv "${DATA_CSV}" \
    --output_root "${EXP_DIR}" \
    --top_n_categories 8 \
    --seed 13
else
  echo "Step 1: Newsgroups data already exists, skipping prep."
fi

# Step 2: Run CPU baselines
echo "Step 2: Running baselines..."
python src/run_baseline_benchmark.py \
  --dataset newsgroups \
  --experiment_dir "${EXP_DIR}" \
  --methods ${METHODS} \
  --n_clusters 8 \
  --text_column description \
  --n_runs "${N_RUNS}" \
  --filter_top_k_categories 8 \
  --output_dir "${EXP_DIR}/baseline_results"

# Free GPU memory from baseline step (SentenceTransformer, BERTopic, etc.)
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true

# Lower GPU memory utilization since baseline step may have fragmented GPU memory
export VLLM_GPU_MEMORY_UTILIZATION=0.70

# Step 3: LLM scoring
echo "Step 3: LLM scoring..."
BASELINE_DIRS=()
for method_dir in "${EXP_DIR}/baseline_results"/*/; do
  if [[ -f "${method_dir}/inner_node_labels.csv" ]]; then
    BASELINE_DIRS+=("${method_dir%/}")
  fi
  for run_dir in "${method_dir}"/run_*/; do
    if [[ -f "${run_dir}/inner_node_labels.csv" ]]; then
      BASELINE_DIRS+=("${run_dir%/}")
    fi
  done
done

if [[ ${#BASELINE_DIRS[@]} -gt 0 ]]; then
  python src/score_baseline_with_llm.py \
    --baseline_dirs "${BASELINE_DIRS[@]}" \
    --sentence_data_path "${DATA_CSV}" \
    --sentence_column description \
    --model_name "${MODEL_NAME}" \
    --model_type vllm \
    --num_trials "${NUM_TRIALS}"
fi

# Step 4: Collate results
echo "Step 4: Collating results..."
python src/collate_baseline_results.py \
  --baseline_root "${EXP_DIR}/baseline_results" \
  --output_prefix "results/baseline_comparison_newsgroups"

echo "========================================"
echo "Finished:    $(date)"
echo "Results at:  results/baseline_comparison_newsgroups.*"
echo "========================================"
