"""
Collate baseline results into comparison tables.

Reads external_metrics.json and llm_scores.json from all baseline
output directories and produces unified comparison CSVs and LaTeX tables.

Usage:
  python src/collate_baseline_results.py \
    --baseline_root experiments/newsgroups_sanity_tiny/baseline_results \
    --output_prefix results/baseline_comparison_newsgroups

  python src/collate_baseline_results.py \
    --baseline_root experiments/wiki_biographies_10000/baseline_results \
    --output_prefix results/baseline_comparison_wiki
"""

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Display names for methods
METHOD_DISPLAY_NAMES = {
    "lda_gibbs": "LDA (Gibbs)",
    "lda_vi": "LDA (VI)",
    "embedding_kmeans": "Embed+KMeans",
    "bertopic": "BERTopic",
    "ctm": "CTM",
    "dvae": "Dirichlet-VAE",
    "scholar": "Scholar",
    "scholar_kd": "Scholar+KD",
    "pipeline_steps1_6": "Steps 1-6",
    "pipeline_em": "Steps 1-6 + EM",
}


def load_method_results(method_dir: str) -> dict:
    """Load all result files from a single method directory."""
    result = {"method_key": os.path.basename(method_dir)}

    # External metrics
    ext_path = os.path.join(method_dir, "external_metrics.json")
    if os.path.exists(ext_path):
        with open(ext_path) as f:
            result.update(json.load(f))

    # LLM scores
    llm_path = os.path.join(method_dir, "llm_scores.json")
    if os.path.exists(llm_path):
        with open(llm_path) as f:
            llm = json.load(f)
            # Prefix LLM keys to avoid collision
            for k, v in llm.items():
                if k not in ("choices", "n_texts", "n_clusters"):
                    result[f"llm_{k}"] = v
                else:
                    result[k] = v

    # Run metadata
    meta_path = os.path.join(method_dir, "run_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
            result["elapsed_seconds"] = meta.get("elapsed_seconds")

    return result


def collate(baseline_root: str) -> pd.DataFrame:
    """Collate results from all method directories under baseline_root."""
    rows = []

    for entry in sorted(os.listdir(baseline_root)):
        method_dir = os.path.join(baseline_root, entry)
        if not os.path.isdir(method_dir):
            continue

        # Check for sub-runs (run_0, run_1, ...)
        sub_runs = [
            d for d in os.listdir(method_dir)
            if os.path.isdir(os.path.join(method_dir, d)) and d.startswith("run_")
        ]

        if sub_runs:
            for run_dir_name in sorted(sub_runs):
                run_dir = os.path.join(method_dir, run_dir_name)
                result = load_method_results(run_dir)
                result["run"] = run_dir_name
                rows.append(result)
        else:
            result = load_method_results(method_dir)
            result["run"] = "run_0"
            rows.append(result)

    if not rows:
        logger.warning("No results found in %s", baseline_root)
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Add display names
    df["method"] = df["method_key"].map(
        lambda k: METHOD_DISPLAY_NAMES.get(k, k)
    )

    return df


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate multiple runs into mean +/- std."""
    if df.empty:
        return df

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # Remove 'run' related columns from aggregation
    group_cols = ["method_key", "method"]

    agg_rows = []
    for (method_key, method), group in df.groupby(group_cols):
        row = {"method_key": method_key, "method": method, "n_runs": len(group)}
        for col in numeric_cols:
            vals = group[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = vals.mean()
                row[f"{col}_std"] = vals.std() if len(vals) > 1 else 0.0
        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


def format_table(agg_df: pd.DataFrame, has_ground_truth: bool = True) -> str:
    """Format as a readable text table."""
    lines = []
    lines.append("=" * 100)
    lines.append("  BASELINE COMPARISON")
    lines.append("=" * 100)

    # Build header
    cols = [("Method", 22)]
    if has_ground_truth:
        cols += [("ARI", 12), ("NMI", 12), ("Purity", 12)]
    cols += [
        ("PPL(asgn)", 12),
        ("PPL(orac)", 12),
        ("avgLL/tok", 12),
        ("K", 4),
    ]

    header = "".join(f"{name:<{w}}" if i == 0 else f"{name:>{w}}" for i, (name, w) in enumerate(cols))
    lines.append(header)
    lines.append("-" * 100)

    for _, row in agg_df.iterrows():
        parts = [f"{row['method']:<22}"]

        if has_ground_truth:
            for metric in ["ARI", "NMI", "Purity"]:
                mean = row.get(f"{metric}_mean")
                std = row.get(f"{metric}_std")
                if mean is not None and not pd.isna(mean):
                    parts.append(f"{mean:>8.4f}±{std:.3f}" if std else f"{mean:>12.4f}")
                else:
                    parts.append(f"{'N/A':>12}")

        for metric in [
            "llm_perplexity_token_assigned",
            "llm_perplexity_token_oracle",
            "llm_avg_logL_per_token_assigned",
        ]:
            mean = row.get(f"{metric}_mean")
            if mean is not None and not pd.isna(mean):
                parts.append(f"{mean:>12.4f}")
            else:
                parts.append(f"{'N/A':>12}")

        k = row.get("n_clusters_mean", row.get("n_clusters"))
        parts.append(f"{int(k):>4}" if k is not None and not pd.isna(k) else f"{'?':>4}")

        lines.append("".join(parts))

    lines.append("=" * 100)
    return "\n".join(lines)


def to_latex(agg_df: pd.DataFrame, has_ground_truth: bool = True) -> str:
    """Generate a LaTeX table."""
    lines = []
    if has_ground_truth:
        cols = "l" + "r" * 6
        header_cells = ["Method", "ARI", "NMI", "Purity", "PPL(asgn)", "PPL(orac)", "K"]
    else:
        cols = "l" + "r" * 3
        header_cells = ["Method", "PPL(asgn)", "PPL(orac)", "K"]

    lines.append(r"\begin{tabular}{" + cols + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    for _, row in agg_df.iterrows():
        cells = [row["method"]]

        if has_ground_truth:
            for metric in ["ARI", "NMI", "Purity"]:
                mean = row.get(f"{metric}_mean")
                if mean is not None and not pd.isna(mean):
                    cells.append(f"{mean:.3f}")
                else:
                    cells.append("--")

        for metric in ["llm_perplexity_token_assigned", "llm_perplexity_token_oracle"]:
            mean = row.get(f"{metric}_mean")
            if mean is not None and not pd.isna(mean):
                cells.append(f"{mean:.2f}")
            else:
                cells.append("--")

        k = row.get("n_clusters_mean", row.get("n_clusters"))
        cells.append(f"{int(k)}" if k is not None and not pd.isna(k) else "?")

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Collate baseline results into comparison tables."
    )
    parser.add_argument(
        "--baseline_root",
        type=str,
        required=True,
        help="Root directory containing baseline method subdirectories.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="results/baseline_comparison",
        help="Prefix for output files (CSV, tex, txt).",
    )
    parser.add_argument(
        "--no_ground_truth",
        action="store_true",
        help="If set, omit ARI/NMI/Purity columns.",
    )

    args = parser.parse_args()
    has_gt = not args.no_ground_truth

    # Collate
    df = collate(args.baseline_root)
    if df.empty:
        logger.error("No results found. Exiting.")
        return

    # Aggregate runs
    agg_df = aggregate_runs(df)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_prefix) or ".", exist_ok=True)

    # Save raw CSV
    csv_path = f"{args.output_prefix}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved raw results: %s", csv_path)

    # Save aggregated CSV
    agg_csv_path = f"{args.output_prefix}_aggregated.csv"
    agg_df.to_csv(agg_csv_path, index=False)
    logger.info("Saved aggregated results: %s", agg_csv_path)

    # Print table
    table = format_table(agg_df, has_gt)
    print(table)

    # Save text table
    txt_path = f"{args.output_prefix}.txt"
    with open(txt_path, "w") as f:
        f.write(table)
    logger.info("Saved text table: %s", txt_path)

    # Save LaTeX table
    tex_path = f"{args.output_prefix}.tex"
    with open(tex_path, "w") as f:
        f.write(to_latex(agg_df, has_gt))
    logger.info("Saved LaTeX table: %s", tex_path)


if __name__ == "__main__":
    main()
