#!/bin/bash
#SBATCH --job-name=step6_agglom_wiki
#SBATCH --account=nlp
#SBATCH --partition=john
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=14-0
#SBATCH --output=logs/step6_agglom_%j.out
#SBATCH --error=logs/step6_agglom_%j.err

set -euo pipefail

REPO_DIR="/nlp/scr/tdalmia/projects/neural-taxonomy"
CENTROIDS_FILE="experiments/wiki_biographies_10000/models/cluster_centroids.npy"
OUTPUT_DIR="experiments/wiki_biographies_10000/hierarchy_results"
CLUSTER_LABELS="experiments/wiki_biographies_10000/models/cluster_labels.csv"
EXAMPLES_CSV="experiments/wiki_biographies_10000/models/all_extracted_discourse_with_clusters.csv"

cd "${REPO_DIR}"
mkdir -p logs

source /nlp/scr/tdalmia/miniconda3/etc/profile.d/conda.sh
conda activate nlp

# Optional: use node-local for temp + compiled extensions (fast), while caches stay on /nlp/scr via hook
if [ -d /scr-ssd ]; then
  LOCAL_SCRATCH=/scr-ssd/$USER
else
  LOCAL_SCRATCH=/nlp/scr/$USER
fi
mkdir -p "$LOCAL_SCRATCH/tmp" "$LOCAL_SCRATCH/torch_extensions"
export TMPDIR="$LOCAL_SCRATCH/tmp"
export TORCH_EXTENSIONS_DIR="$LOCAL_SCRATCH/torch_extensions"

echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "HF_HOME=$HF_HOME"
echo "MPLCONFIGDIR=$MPLCONFIGDIR"
echo "TMPDIR=$TMPDIR"

# NOTE: step 6 labeling uses OpenAI by default; set OPENAI_API_KEY in env before sbatch if needed.
python src/step_6__agglomerative_clustering.py \
  "${CENTROIDS_FILE}" \
  "${OUTPUT_DIR}" \
  --min_clusters 2 \
  --max_clusters 10 \
  --min_cluster_size 2 \
  --min_hierarchical_levels 2 \
  --method ward \
  --metric euclidean \
  --label_tree \
  --initial_cluster_labels_path "${CLUSTER_LABELS}" \
  --min_descendants 0 \
  --max_depth 8 \
  --num_samples_per_node_for_labeling 5 \
  --raw_data_examples_df_path "${EXAMPLES_CSV}" \
  --examples_cluster_col cluster \
  --examples_sentence_col description \
  --no_visualize
