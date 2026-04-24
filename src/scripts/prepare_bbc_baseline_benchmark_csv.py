"""
Merge BBC raw articles (ground truth) with step-4 output so run_baseline_benchmark.py
can load document text + true_label from the standard path:

  <experiment_dir>/models/all_extracted_discourse_with_clusters_and_text.csv

Step 4 only has LLM fields + cluster; raw text lives in data/bbc_news.csv.
Join on step-4 ``index`` == raw ``id`` (both stringified).

Usage:
  python src/scripts/prepare_bbc_baseline_benchmark_csv.py \\
      --experiment_dir experiments/bbc_news

Then:
  python src/run_baseline_benchmark.py \\
      --dataset bbc \\
      --experiment_dir experiments/bbc_news \\
      --n_clusters 5 \\
      --text_column text \\
      --methods lda_gibbs embedding_kmeans bertopic
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiments/bbc_news",
        help="BBC experiment root (contains data/ and models/).",
    )
    parser.add_argument(
        "--raw_csv",
        type=str,
        default=None,
        help="Override path to bbc_news.csv (default: <experiment_dir>/data/bbc_news.csv).",
    )
    parser.add_argument(
        "--step4_csv",
        type=str,
        default=None,
        help="Override step-4 CSV (default: <experiment_dir>/models/all_extracted_discourse_with_clusters.csv).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output path (default: <experiment_dir>/models/all_extracted_discourse_with_clusters_and_text.csv).",
    )
    args = parser.parse_args()

    exp = Path(args.experiment_dir)
    raw_path = Path(args.raw_csv or exp / "data" / "bbc_news.csv")
    step4_path = Path(
        args.step4_csv
        or exp / "models" / "all_extracted_discourse_with_clusters.csv"
    )
    out_path = Path(
        args.output_csv
        or exp / "models" / "all_extracted_discourse_with_clusters_and_text.csv"
    )

    if not raw_path.is_file():
        raise FileNotFoundError(f"Missing raw data: {raw_path}")
    if not step4_path.is_file():
        raise FileNotFoundError(f"Missing step-4 output: {step4_path}")

    raw_df = pd.read_csv(raw_path)
    if "id" not in raw_df.columns or "text" not in raw_df.columns:
        raise KeyError(f"{raw_path} must have columns id, text (and true_label for metrics).")
    if "true_label" not in raw_df.columns:
        logging.warning("No true_label in raw CSV; baselines will not compute ARI/NMI/Purity.")

    s4 = pd.read_csv(step4_path)
    if "index" not in s4.columns:
        raise KeyError(f"{step4_path} must have column index (document id from step 1).")

    raw_df = raw_df.copy()
    raw_df["id"] = raw_df["id"].astype(str)
    s4 = s4.copy()
    s4["index"] = s4["index"].astype(str)

    merged = s4.merge(
        raw_df[["id", "text", "true_label"]],
        left_on="index",
        right_on="id",
        how="inner",
    )
    if len(merged) < len(s4):
        logging.warning(
            "Dropped %d step-4 rows with no matching raw id.",
            len(s4) - len(merged),
        )
    if len(merged) < len(raw_df):
        logging.warning(
            "%d raw rows had no step-4 row (ok if you subsampled the pipeline).",
            len(raw_df) - len(merged),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    logging.info("Wrote %d rows to %s", len(merged), out_path)
    logging.info("Columns: %s", list(merged.columns))


if __name__ == "__main__":
    main()
