"""
Build a LaTeX table of BBC News: flat baselines (mean ± std over runs) + pipeline hierarchy.

Reads:
  experiments/bbc_news/baseline_results/baseline_summary.csv
  experiments/bbc_news/evaluation_results.csv
  experiments/bbc_news/models/all_extracted_discourse_with_clusters_and_text.csv (for step-4 vs GT)

Usage:
  python src/scripts/make_bbc_results_table.py \\
      --experiment_dir experiments/bbc_news \\
      --out experiments/bbc_news/bbc_results_table.tex

  # print to stdout only:
  python src/scripts/make_bbc_results_table.py --experiment_dir experiments/bbc_news
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def _alignment_metrics(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> dict:
    cont = contingency_matrix(ground_truth, cluster_labels)
    purity = float(np.sum(np.amax(cont, axis=0)) / np.sum(cont))
    return {
        "ARI": float(adjusted_rand_score(ground_truth, cluster_labels)),
        "NMI": float(
            normalized_mutual_info_score(
                ground_truth, cluster_labels, average_method="arithmetic"
            )
        ),
        "Purity": purity,
    }


def _fmt_pm(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"


def _fmt_single(x: float, decimals: int = 3) -> str:
    return f"{x:.{decimals}f}"


def baseline_section(summary: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    order = [
        ("embedding_kmeans", "Embed+KMeans (MiniLM)"),
        ("lda_gibbs", "LDA (Gibbs / tomotopy)"),
        ("lda_vi", "LDA (VI / sklearn)"),
        ("bertopic", "BERTopic"),
    ]
    for key, display in order:
        sub = summary.loc[summary["method"] == key]
        if sub.empty:
            continue
        k = int(sub["n_clusters"].iloc[0])
        ari_m, ari_s = sub["ARI"].mean(), sub["ARI"].std(ddof=1)
        nmi_m, nmi_s = sub["NMI"].mean(), sub["NMI"].std(ddof=1)
        pur_m, pur_s = sub["Purity"].mean(), sub["Purity"].std(ddof=1)
        lines.append(
            f"{display} & {k} & "
            f"${_fmt_pm(ari_m, ari_s)}$ & ${_fmt_pm(nmi_m, nmi_s)}$ & ${_fmt_pm(pur_m, pur_s)}$ \\\\"
        )
    return lines


def pipeline_section(eval_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    lines.append(
        r"\midrule"
        + "\n"
        + r"\multicolumn{5}{l}{\emph{Pipeline (steps 1--6), hierarchical levels vs.\ five BBC categories}} \\"
    )
    for _, row in eval_df.iterrows():
        lev = int(row["level"])
        k = int(row["n_clusters"])
        ari, nmi, pur = row["ARI"], row["NMI"], row["Purity"]
        lines.append(
            f"Steps 1--6 (level {lev}) & {k} & "
            f"${_fmt_single(float(ari))}$ & ${_fmt_single(float(nmi))}$ & ${_fmt_single(float(pur))}$ \\\\"
        )
    return lines


def step4_row(merged_csv: Path) -> str | None:
    if not merged_csv.is_file():
        return None
    df = pd.read_csv(merged_csv)
    if "cluster" not in df.columns or "true_label" not in df.columns:
        return None
    y_pred = df["cluster"].astype(int).values
    y_true = df["true_label"].astype(str).values
    m = _alignment_metrics(y_pred, y_true)
    k = int(df["cluster"].nunique())
    return (
        r"\midrule"
        + "\n"
        + r"\multicolumn{5}{l}{\emph{Pipeline step 4: FT-SBERT embeddings + K-means micro-clusters vs.\ GT}} \\"
        + "\n"
        f"Steps 1--4 (FT-SBERT + K-means) & {k} & "
        f"${_fmt_single(m['ARI'])}$ & ${_fmt_single(m['NMI'])}$ & ${_fmt_single(m['Purity'])}$ \\\\"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, default="experiments/bbc_news")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Write LaTeX here (default: print only).",
    )
    args = parser.parse_args()

    exp = Path(args.experiment_dir)
    summary_path = exp / "baseline_results" / "baseline_summary.csv"
    eval_path = exp / "evaluation_results.csv"
    merged = exp / "models" / "all_extracted_discourse_with_clusters_and_text.csv"

    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    summary = pd.read_csv(summary_path)
    eval_df = pd.read_csv(eval_path)

    n_docs = int(eval_df["n_docs_evaluated"].iloc[0])
    n_gt = 5

    header = [
        r"\begin{center}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        rf"Method & $K$ & ARI & NMI & Purity \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\emph{Flat baselines ($K=5$ topics except BERTopic effective $K$); mean $\pm$ std.\ over 3 seeds}} \\",
    ]

    body_baseline = baseline_section(summary)
    s4 = step4_row(merged)
    body_pipe = pipeline_section(eval_df)

    footer = [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{center}",
        rf"\footnotesize $N={n_docs}$ documents; {n_gt} ground-truth categories (SetFit/BBC). NPMI not computed in this repo.",
        r"\normalsize",
    ]

    lines = (
        [rf"% Auto-generated table for {exp.as_posix()}", ""]
        + header
        + body_baseline
        + ([s4] if s4 else [])
        + body_pipe
        + footer
    )
    text = "\n".join(lines) + "\n"

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
