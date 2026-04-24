"""
Baseline Benchmark Runner for Neural Taxonomy Evaluation.

Runs standard topic model baselines on the project's datasets
(newsgroups, wiki biographies) and writes outputs in a format
compatible with the EM pipeline's scoring infrastructure.

Usage:
  python src/run_baseline_benchmark.py \
    --dataset newsgroups \
    --experiment_dir experiments/newsgroups_sanity_tiny \
    --methods lda_gibbs embedding_kmeans bertopic \
    --n_clusters 8 \
    --text_column description \
    --output_dir experiments/newsgroups_sanity_tiny/baseline_results

  python src/run_baseline_benchmark.py \
    --dataset wiki \
    --experiment_dir experiments/wiki_biographies_10000 \
    --methods lda_gibbs embedding_kmeans bertopic \
    --n_clusters 10 \
    --text_column sentence_text \
    --output_dir experiments/wiki_biographies_10000/baseline_results

  # BBC (after pipeline + prepare_bbc_baseline_benchmark_csv.py):
  python src/run_baseline_benchmark.py \
    --dataset bbc \
    --experiment_dir experiments/bbc_news \
    --methods lda_gibbs embedding_kmeans bertopic \
    --n_clusters 5 \
    --text_column text
"""

import argparse
import json
import logging
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reuse existing baseline implementations
from run_all_baselines import (
    METHODS as BASELINE_METHODS,
    compute_latent_perplexity,
)

# Reuse existing evaluation metrics
from eval_topic_model_metrics import compute_ari, compute_nmi, compute_purity

# Reuse existing label induction
from schema_utils import (
    propose_label_from_texts,
    propose_label_and_keywords_from_texts,
    build_label_string,
    sanitize_text,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_project_data(
    dataset: str,
    experiment_dir: str,
    text_column: str = "description",
    filter_top_k_categories: Optional[int] = None,
) -> pd.DataFrame:
    """Load data from the project's CSV format.

    For newsgroups: CSV with centroid_id, description, true_label
    For wiki: CSV with custom_id, sentence_text, cluster, label, etc.
    For bbc: use ``prepare_bbc_baseline_benchmark_csv.py`` to write
    ``models/all_extracted_discourse_with_clusters_and_text.csv`` (text + true_label).

    filter_top_k_categories: if set, filter rows to only the K most common
    true_label values. This is needed for newsgroups where the raw CSV has all
    20 categories but the EM pipeline only operates on the top-K filtered subset.
    """
    data_path = os.path.join(
        experiment_dir, "models", "all_extracted_discourse_with_clusters_and_text.csv"
    )
    if not os.path.exists(data_path):
        # Fallback for wiki_biographies (older path)
        data_path = os.path.join(
            experiment_dir, "models", "all_extracted_discourse_with_clusters.csv"
        )

    logger.info("Loading data from: %s", data_path)
    df = pd.read_csv(data_path)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))

    # Validate text column exists
    if text_column not in df.columns:
        available = [c for c in df.columns if "text" in c.lower() or "desc" in c.lower()]
        raise ValueError(
            f"Text column '{text_column}' not found. Available text-like columns: {available}"
        )

    # For newsgroups, true_label column exists
    if "true_label" in df.columns:
        n_cats_raw = df["true_label"].nunique()
        logger.info("Ground truth labels found. Categories: %d", n_cats_raw)

        # Filter to top-K categories to match what the EM pipeline operates on.
        # The raw CSV has all 20 newsgroups but the EM hierarchy only uses top-K.
        if filter_top_k_categories is not None and filter_top_k_categories < n_cats_raw:
            top_cats = (
                df["true_label"]
                .value_counts()
                .head(filter_top_k_categories)
                .index.tolist()
            )
            df = df[df["true_label"].isin(top_cats)].reset_index(drop=True)
            logger.info(
                "Filtered to top-%d categories: %s (%d rows remaining)",
                filter_top_k_categories,
                top_cats,
                len(df),
            )

    return df


# ---------------------------------------------------------------------------
# Label induction for baseline clusters
# ---------------------------------------------------------------------------

def induce_cluster_labels(
    clusters: np.ndarray,
    texts: List[str],
    max_keywords: int = 4,
) -> Dict[int, str]:
    """Generate text labels for each cluster using TF-IDF keywords.

    Returns: {cluster_id: "label string"}
    """
    unique_clusters = sorted(set(clusters))
    label_map = {}

    for cid in unique_clusters:
        mask = clusters == cid
        cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]

        if not cluster_texts:
            label_map[cid] = f"Cluster_{cid}"
            continue

        name, keywords = propose_label_and_keywords_from_texts(
            cluster_texts, max_keywords=max_keywords
        )
        label = build_label_string(name, keywords, include_keywords=True)
        label_map[cid] = label

    # Deduplicate labels
    label_counts = Counter(label_map.values())
    for cid, label in list(label_map.items()):
        if label_counts[label] > 1:
            label_map[cid] = f"{label} ({cid})"

    return label_map


# ---------------------------------------------------------------------------
# Centroid-level aggregation
# ---------------------------------------------------------------------------

def aggregate_to_centroid_level(
    df: pd.DataFrame,
    clusters: np.ndarray,
    centroid_col: str = "centroid_id",
) -> pd.DataFrame:
    """Map document-level cluster assignments to centroid level via majority vote.

    Returns DataFrame with centroid_id, cluster_id (majority), cluster_count.
    """
    if centroid_col not in df.columns:
        # No centroid column: treat each row as its own centroid
        return pd.DataFrame({
            "centroid_id": np.arange(len(clusters)),
            "cluster_id": clusters,
        })

    df_work = df[[centroid_col]].copy()
    df_work["cluster_id"] = clusters

    # Majority vote per centroid
    centroid_clusters = (
        df_work.groupby(centroid_col)["cluster_id"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
    )
    centroid_clusters.columns = ["centroid_id", "cluster_id"]
    return centroid_clusters


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_baseline_output(
    output_dir: str,
    centroid_clusters: pd.DataFrame,
    label_map: Dict[int, str],
    doc_assignments: pd.DataFrame,
    external_metrics: Optional[Dict] = None,
    run_metadata: Optional[Dict] = None,
):
    """Write baseline results in pipeline-compatible format.

    Creates:
      - clusters_level_1_K_clusters.csv  (centroid-to-cluster mapping)
      - inner_node_labels.csv            (cluster text labels)
      - doc_cluster_assignments.csv      (full doc-level assignments)
      - external_metrics.json            (ARI, NMI, Purity if available)
      - run_metadata.json                (method, params, runtime)
    """
    os.makedirs(output_dir, exist_ok=True)

    # clusters_level_1_K_clusters.csv
    # Format: centroid_id, fcluster_label, graph_node_id
    n_clusters = len(label_map)
    cluster_ids = sorted(label_map.keys())
    # Map cluster_id to sequential graph_node_id
    cid_to_gid = {cid: gid for gid, cid in enumerate(cluster_ids)}

    clusters_df = pd.DataFrame({
        "centroid_id": centroid_clusters["centroid_id"],
        "fcluster_label": centroid_clusters["cluster_id"].map(
            lambda x: cid_to_gid.get(x, 0) + 1
        ),
        "graph_node_id": centroid_clusters["cluster_id"].map(
            lambda x: n_clusters + cid_to_gid.get(x, 0)
        ),
    })
    clusters_path = os.path.join(
        output_dir, f"clusters_level_1_{n_clusters}_clusters.csv"
    )
    clusters_df.to_csv(clusters_path, index=False)
    logger.info("Wrote centroid clusters: %s", clusters_path)

    # inner_node_labels.csv
    # Format: node_id, label, description
    labels_df = pd.DataFrame([
        {
            "node_id": n_clusters + cid_to_gid[cid],
            "label": label_map[cid],
            "description": label_map[cid],
        }
        for cid in cluster_ids
    ])
    labels_path = os.path.join(output_dir, "inner_node_labels.csv")
    labels_df.to_csv(labels_path, index=False)
    logger.info("Wrote cluster labels: %s", labels_path)

    # optimal_thresholds.csv (dummy, for compatibility)
    thresholds_df = pd.DataFrame([{
        "threshold": 1.0,
        "n_clusters": n_clusters,
        "silhouette_score": 0.0,
    }])
    thresholds_path = os.path.join(output_dir, "optimal_thresholds.csv")
    thresholds_df.to_csv(thresholds_path, index=False)

    # doc_cluster_assignments.csv
    doc_path = os.path.join(output_dir, "doc_cluster_assignments.csv")
    doc_assignments.to_csv(doc_path, index=False)
    logger.info("Wrote doc assignments: %s (%d rows)", doc_path, len(doc_assignments))

    # external_metrics.json
    if external_metrics:
        metrics_path = os.path.join(output_dir, "external_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(external_metrics, f, indent=2)
        logger.info("Wrote external metrics: %s", metrics_path)

    # run_metadata.json
    if run_metadata:
        meta_path = os.path.join(output_dir, "run_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(run_metadata, f, indent=2, default=str)
        logger.info("Wrote metadata: %s", meta_path)


# ---------------------------------------------------------------------------
# Single method runner
# ---------------------------------------------------------------------------

def run_single_baseline(
    method_key: str,
    texts: List[str],
    n_clusters: int,
    ground_truth: Optional[np.ndarray],
    random_state: int = 42,
) -> Dict:
    """Run a single baseline method and compute metrics.

    Returns dict with: clusters, doc_topic_dist, label_map, external_metrics, runtime.
    """
    method_info = BASELINE_METHODS[method_key]
    method_name = method_info["name"]

    logger.info("Running %s (K=%d, seed=%d)...", method_name, n_clusters, random_state)
    start_time = time.time()

    try:
        clusters, doc_topic_dist = method_info["fn"](texts, n_clusters, random_state)
        elapsed = time.time() - start_time
        logger.info("  %s completed in %.1fs", method_name, elapsed)
    except Exception as e:
        logger.error("  %s FAILED: %s", method_name, e)
        return {"error": str(e), "method": method_name}

    clusters = np.asarray(clusters)

    # Induce text labels
    label_map = induce_cluster_labels(clusters, texts)

    # External metrics (if ground truth available)
    external_metrics = None
    if ground_truth is not None:
        external_metrics = {
            "ARI": float(compute_ari(clusters, ground_truth)),
            "NMI": float(compute_nmi(clusters, ground_truth)),
            "Purity": float(compute_purity(clusters, ground_truth)),
            "n_clusters": int(len(set(clusters))),
            "n_true_classes": int(len(set(ground_truth))),
        }
        logger.info(
            "  ARI=%.4f  NMI=%.4f  Purity=%.4f",
            external_metrics["ARI"],
            external_metrics["NMI"],
            external_metrics["Purity"],
        )

    # Latent perplexity
    latent_ppl = None
    if doc_topic_dist is not None and doc_topic_dist.ndim == 2:
        latent_ppl = float(compute_latent_perplexity(doc_topic_dist))
        if external_metrics:
            external_metrics["latent_perplexity"] = latent_ppl
        logger.info("  Latent perplexity: %.2f", latent_ppl)

    return {
        "method": method_name,
        "method_key": method_key,
        "clusters": clusters,
        "doc_topic_dist": doc_topic_dist,
        "label_map": label_map,
        "external_metrics": external_metrics,
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset: str,
    experiment_dir: str,
    methods: List[str],
    n_clusters: int,
    text_column: str,
    output_dir: str,
    n_runs: int = 1,
    filter_top_k_categories: Optional[int] = None,
):
    """Run all specified baselines and write outputs."""
    # Load data
    df = load_project_data(dataset, experiment_dir, text_column, filter_top_k_categories)
    texts = df[text_column].astype(str).tolist()

    # Ground truth (if available)
    ground_truth = None
    if "true_label" in df.columns:
        ground_truth = df["true_label"].values

    # Centroid column (if available)
    centroid_col = "centroid_id" if "centroid_id" in df.columns else None

    logger.info(
        "Benchmark: dataset=%s, n_texts=%d, n_clusters=%d, methods=%s, n_runs=%d",
        dataset, len(texts), n_clusters, methods, n_runs,
    )

    all_results = []

    for method_key in methods:
        if method_key not in BASELINE_METHODS:
            logger.warning("Unknown method: %s (skipping)", method_key)
            continue

        for run_idx in range(n_runs):
            random_state = 42 + run_idx
            run_tag = f"run_{run_idx}" if n_runs > 1 else ""

            result = run_single_baseline(
                method_key, texts, n_clusters, ground_truth, random_state
            )

            if "error" in result:
                all_results.append({
                    "method": method_key,
                    "run": run_idx,
                    "error": result["error"],
                })
                continue

            # Build doc-level assignments table
            doc_df = pd.DataFrame({
                "doc_id": range(len(texts)),
                "text": texts,
                "cluster_id": result["clusters"],
                "cluster_label": [
                    result["label_map"].get(c, f"Cluster_{c}")
                    for c in result["clusters"]
                ],
            })
            if ground_truth is not None:
                doc_df["true_label"] = ground_truth

            # Aggregate to centroid level
            centroid_clusters = aggregate_to_centroid_level(
                df, result["clusters"], centroid_col or "centroid_id"
            )

            # Output directory
            method_dir = os.path.join(output_dir, method_key)
            if run_tag:
                method_dir = os.path.join(method_dir, run_tag)

            # Write outputs
            write_baseline_output(
                output_dir=method_dir,
                centroid_clusters=centroid_clusters,
                label_map=result["label_map"],
                doc_assignments=doc_df,
                external_metrics=result.get("external_metrics"),
                run_metadata={
                    "method": result["method"],
                    "method_key": method_key,
                    "dataset": dataset,
                    "n_clusters": n_clusters,
                    "n_texts": len(texts),
                    "random_state": random_state,
                    "run_index": run_idx,
                    "elapsed_seconds": result["elapsed_seconds"],
                    "text_column": text_column,
                    "experiment_dir": experiment_dir,
                },
            )

            all_results.append({
                "method": method_key,
                "method_name": result["method"],
                "run": run_idx,
                "output_dir": method_dir,
                **(result.get("external_metrics") or {}),
                "elapsed_seconds": result["elapsed_seconds"],
            })

    # Print summary
    print_summary(all_results, dataset)

    # Save summary CSV
    summary_path = os.path.join(output_dir, "baseline_summary.csv")
    pd.DataFrame(all_results).to_csv(summary_path, index=False)
    logger.info("Summary saved to: %s", summary_path)

    return all_results


def print_summary(results: List[Dict], dataset: str):
    """Print a formatted summary table."""
    print(f"\n{'=' * 80}")
    print(f"  BASELINE RESULTS: {dataset.upper()}")
    print(f"{'=' * 80}")

    has_gt = any(r.get("ARI") is not None for r in results)

    if has_gt:
        header = f"{'Method':<20} {'Run':>4} {'ARI':>8} {'NMI':>8} {'Purity':>8} {'Time':>8}"
    else:
        header = f"{'Method':<20} {'Run':>4} {'K':>4} {'Time':>8}"
    print(header)
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['method']:<20} {r['run']:>4}  FAILED: {r['error']}")
            continue

        if has_gt:
            print(
                f"{r.get('method_name', r['method']):<20} {r['run']:>4} "
                f"{r.get('ARI', 0):>8.4f} {r.get('NMI', 0):>8.4f} "
                f"{r.get('Purity', 0):>8.4f} {r['elapsed_seconds']:>7.1f}s"
            )
        else:
            print(
                f"{r.get('method_name', r['method']):<20} {r['run']:>4} "
                f"{r.get('n_clusters', '?'):>4} {r['elapsed_seconds']:>7.1f}s"
            )

    print(f"{'=' * 80}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline topic model benchmarks on project datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["newsgroups", "wiki", "bbc"],
        required=True,
        help="Dataset label (bbc: same CSV layout as newsgroups loader under experiment_dir).",
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory (e.g. experiments/newsgroups_sanity_tiny).",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["lda_gibbs", "lda_vi", "embedding_kmeans", "bertopic"],
        help=f"Baseline methods to run. Options: {list(BASELINE_METHODS.keys())}",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        required=True,
        help="Number of clusters (e.g. 8 newsgroups, 10 wiki, 5 BBC categories).",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="description",
        help="Column name for text data (default: description).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: <experiment_dir>/baseline_results).",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs with different random seeds (default: 1).",
    )
    parser.add_argument(
        "--filter_top_k_categories",
        type=int,
        default=None,
        help=(
            "If set, filter data to only the K most common true_label values before "
            "running baselines. Use this for newsgroups to match the EM pipeline's "
            "top-K category filtering (e.g. --filter_top_k_categories 8)."
        ),
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.experiment_dir, "baseline_results")

    run_benchmark(
        dataset=args.dataset,
        experiment_dir=args.experiment_dir,
        methods=args.methods,
        n_clusters=args.n_clusters,
        text_column=args.text_column,
        output_dir=args.output_dir,
        n_runs=args.n_runs,
        filter_top_k_categories=args.filter_top_k_categories,
    )


if __name__ == "__main__":
    main()
