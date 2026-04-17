"""
Evaluate the 6-step pipeline output against ground-truth labels.

Joins the agglomerative cluster assignments from step 6 with the original
document-level data (which has true_label), then computes NMI, ARI, and Purity
at each hierarchical level.

Usage:
  python src/scripts/evaluate_pipeline.py \
      --experiment_dir experiments/bbc_news \
      --raw_data_csv experiments/bbc_news/data/bbc_news.csv \
      --true_label_col true_label
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval_topic_model_metrics import compute_all_alignment_metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def resolve_hierarchy_results_dir(experiment_dir: Path) -> Path:
    """
    Step 6 writes to ``{output_dir}__{variant}`` (e.g. hierarchy_results__standard).
    Users often ``mkdir -p hierarchy_results`` before step 6, leaving an empty
    directory that must not shadow the real outputs.
    """
    default_dir = experiment_dir / "hierarchy_results"
    default_thresh = default_dir / "optimal_thresholds.csv"
    if default_thresh.is_file():
        return default_dir

    candidates: list[Path] = []
    for d in sorted(experiment_dir.glob("hierarchy_results__*")):
        t = d / "optimal_thresholds.csv"
        if t.is_file():
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No optimal_thresholds.csv under {default_dir} or "
            f"{experiment_dir}/hierarchy_results__*"
        )
    candidates.sort(
        key=lambda p: (p / "optimal_thresholds.csv").stat().st_mtime,
        reverse=True,
    )
    chosen = candidates[0]
    logging.info("Using hierarchy dir: %s", chosen)
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--raw_data_csv", type=str, required=True,
                        help="Original CSV with id and true_label columns.")
    parser.add_argument("--true_label_col", type=str, default="true_label")
    parser.add_argument("--id_col", type=str, default="id",
                        help="ID column in the raw data CSV.")
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    models_dir = experiment_dir / "models"

    hierarchy_dir = resolve_hierarchy_results_dir(experiment_dir)

    # Load ground truth
    raw_df = pd.read_csv(args.raw_data_csv)
    if args.true_label_col not in raw_df.columns:
        raise KeyError(f"Column '{args.true_label_col}' not found in {args.raw_data_csv}")
    logging.info("Loaded %d rows of ground truth from %s", len(raw_df), args.raw_data_csv)

    # Load step-4 output (maps each document/row to a centroid_id via 'cluster')
    step4_path = models_dir / "all_extracted_discourse_with_clusters.csv"
    if not step4_path.exists():
        raise FileNotFoundError(f"Step 4 output not found: {step4_path}")
    step4_df = pd.read_csv(step4_path)
    if "cluster" in step4_df.columns:
        step4_df = step4_df.rename(columns={"cluster": "centroid_id"})
    logging.info("Loaded %d rows from step 4 output.", len(step4_df))

    # Load the optimal thresholds to find available hierarchical levels
    thresholds_path = hierarchy_dir / "optimal_thresholds.csv"
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Optimal thresholds not found: {thresholds_path}")
    thresholds_df = pd.read_csv(thresholds_path)
    if "level" not in thresholds_df.columns:
        thresholds_df["level"] = range(1, len(thresholds_df) + 1)

    logging.info("Found %d hierarchical levels.", len(thresholds_df))

    # Evaluate at each level
    results = []
    for _, row in thresholds_df.iterrows():
        level = int(row["level"])
        n_clusters = int(row["n_clusters"])
        cluster_file = hierarchy_dir / f"clusters_level_{level}_{n_clusters}_clusters.csv"

        if not cluster_file.exists():
            logging.warning("Cluster file not found, skipping: %s", cluster_file)
            continue

        level_df = pd.read_csv(cluster_file)

        # Join: step4 rows → agglomerative cluster via centroid_id
        merged = step4_df.merge(level_df[["centroid_id", "fcluster_label"]], on="centroid_id", how="inner")

        # Attach ground truth via row position (step 1 preserves input order;
        # the custom_id / index column carries the original id).
        if "custom_id" in merged.columns:
            id_col_in_merged = "custom_id"
        elif "index" in merged.columns:
            id_col_in_merged = "index"
        else:
            id_col_in_merged = None

        if id_col_in_merged is not None:
            merged[id_col_in_merged] = merged[id_col_in_merged].astype(str)
            raw_df[args.id_col] = raw_df[args.id_col].astype(str)
            eval_df = merged.merge(
                raw_df[[args.id_col, args.true_label_col]],
                left_on=id_col_in_merged,
                right_on=args.id_col,
                how="inner",
            )
        else:
            # Fallback: assume row order alignment (step 4 preserves input order)
            logging.warning("No ID column found for join; falling back to positional alignment.")
            min_len = min(len(merged), len(raw_df))
            eval_df = merged.iloc[:min_len].copy()
            eval_df[args.true_label_col] = raw_df[args.true_label_col].iloc[:min_len].values

        # Drop rows without ground truth
        eval_df = eval_df.dropna(subset=[args.true_label_col, "fcluster_label"])

        if len(eval_df) == 0:
            logging.warning("Level %d: no matched rows after join, skipping.", level)
            continue

        metrics = compute_all_alignment_metrics(
            cluster_labels=eval_df["fcluster_label"].values,
            ground_truth=eval_df[args.true_label_col].values,
        )
        metrics["level"] = level
        metrics["n_clusters"] = n_clusters
        metrics["n_docs_evaluated"] = len(eval_df)
        results.append(metrics)

        logging.info(
            "Level %d (%d clusters, %d docs): ARI=%.4f  NMI=%.4f  Purity=%.4f",
            level, n_clusters, len(eval_df),
            metrics["ARI"], metrics["NMI"], metrics["Purity"],
        )

    if not results:
        logging.error("No levels could be evaluated.")
        return

    results_df = pd.DataFrame(results)
    out_path = experiment_dir / "evaluation_results.csv"
    results_df.to_csv(out_path, index=False)
    logging.info("Saved evaluation results to %s", out_path)

    # Print summary
    best = results_df.loc[results_df["NMI"].idxmax()]
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nBest NMI at level {int(best['level'])} ({int(best['n_clusters'])} clusters):")
    print(f"  ARI    = {best['ARI']:.4f}")
    print(f"  NMI    = {best['NMI']:.4f}")
    print(f"  Purity = {best['Purity']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
