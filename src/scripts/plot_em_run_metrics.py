#!/usr/bin/env python3
"""
Plot EM run metrics from em_iteration_metrics_summary.csv files.

This script accepts either:
1) A single run directory containing:
   <run_dir>/em_diagnostics/em_iteration_metrics_summary.csv
2) A parent directory containing multiple run folders, each with:
   <run_dir>/em_diagnostics/em_iteration_metrics_summary.csv

Outputs:
- Per-run plot: em_diagnostics/em_metrics_over_time.png
  - Includes iteration 0 (pre-change state) when pre_* columns are available.
  - X-axis labels include operation and acceptance per iteration (e.g., split:ok, add:rej).
- Multi-run comparison (if multiple runs): <output_dir>/em_runs_comparison.png
- Multi-run summary table (if multiple runs): <output_dir>/em_runs_summary.csv

Example:
  python src/scripts/plot_em_run_metrics.py \
    --input_dir experiments/wiki_biographies_10000/em_runs_structural
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class RunMetrics:
    run_dir: Path
    metrics_csv: Path
    df: pd.DataFrame


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _parse_accepted(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(True)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "t"})


def _augment_with_iter0(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add an iteration-0 row (pre-change state) when pre_* columns are available.
    Iteration 0 is inferred from the first row's pre_* metrics.
    """
    if df.empty:
        return df
    if "iter" not in df.columns:
        return df

    first = df.iloc[0]
    iter0 = {}
    for col in df.columns:
        iter0[col] = np.nan
    iter0["iter"] = 0
    iter0["selected_operation"] = "init"
    iter0["accepted"] = True

    # Lift any available pre_* metric into iteration 0.
    for col in df.columns:
        pre_col = f"pre_{col}"
        if pre_col in df.columns and pd.notna(first.get(pre_col)):
            iter0[col] = first[pre_col]

    # Special-case common metric aliases.
    if pd.isna(iter0.get("avg_logL_per_token_assigned")) and "pre_avg_logL_per_token_assigned" in df.columns:
        iter0["avg_logL_per_token_assigned"] = first["pre_avg_logL_per_token_assigned"]
    if pd.isna(iter0.get("PPL_tok_assigned")) and "pre_PPL_tok_assigned" in df.columns:
        iter0["PPL_tok_assigned"] = first["pre_PPL_tok_assigned"]
    if pd.isna(iter0.get("avg_logL_per_token_oracle")) and "pre_avg_logL_per_token_oracle" in df.columns:
        iter0["avg_logL_per_token_oracle"] = first["pre_avg_logL_per_token_oracle"]
    if pd.isna(iter0.get("PPL_tok_oracle")) and "pre_PPL_tok_oracle" in df.columns:
        iter0["PPL_tok_oracle"] = first["pre_PPL_tok_oracle"]

    # Infer pre-change cluster count from first iteration action counts:
    # post = pre + splits + adds - merges - removes  => pre = post - splits - adds + merges + removes
    if "clusters" in df.columns and pd.notna(first.get("clusters")):
        splits = float(first.get("splits", 0.0) or 0.0)
        adds = float(first.get("adds", 0.0) or 0.0)
        merges = float(first.get("merges", 0.0) or 0.0)
        removes = float(first.get("removes", 0.0) or 0.0)
        iter0["clusters"] = float(first["clusters"]) - splits - adds + merges + removes

    iter0_df = pd.DataFrame([iter0])
    out = pd.concat([iter0_df, df], axis=0, ignore_index=True)
    out = out.sort_values("iter").reset_index(drop=True)
    return out


def _iteration_tick_labels(df: pd.DataFrame) -> list[str]:
    labels: list[str] = []
    accepted = _parse_accepted(df["accepted"]) if "accepted" in df.columns else pd.Series([True] * len(df))
    for idx, row in df.iterrows():
        it = int(row["iter"])
        if it == 0:
            labels.append("0\ninit")
            continue
        op = str(row.get("selected_operation", "none"))
        if op == "nan":
            op = "none"
        status = "ok" if bool(accepted.iloc[idx]) else "rej"
        labels.append(f"{it}\n{op}:{status}")
    return labels


def _find_metrics_csvs(input_dir: Path, recursive: bool = False) -> list[Path]:
    direct = input_dir / "em_diagnostics" / "em_iteration_metrics_summary.csv"
    if direct.exists():
        return [direct]

    child_paths = sorted(input_dir.glob("*/em_diagnostics/em_iteration_metrics_summary.csv"))
    if child_paths and not recursive:
        return child_paths

    recursive_paths = sorted(input_dir.rglob("em_iteration_metrics_summary.csv"))
    return [p for p in recursive_paths if p.parent.name == "em_diagnostics"]


def _load_runs(metrics_csvs: list[Path]) -> list[RunMetrics]:
    runs: list[RunMetrics] = []
    for csv_path in metrics_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[warn] failed to read {csv_path}: {exc}")
            continue

        if df.empty:
            print(f"[warn] empty metrics file: {csv_path}")
            continue

        if "iter" in df.columns:
            df = df.sort_values("iter").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            df["iter"] = np.arange(1, len(df) + 1, dtype=int)

        df_aug = _augment_with_iter0(df)
        run_dir = csv_path.parent.parent
        runs.append(RunMetrics(run_dir=run_dir, metrics_csv=csv_path, df=df_aug))
    return runs


def _plot_single_run(run: RunMetrics, dpi: int = 170) -> Path:
    df = run.df
    x = df["iter"].astype(int).to_numpy()
    accepted = _parse_accepted(df["accepted"]) if "accepted" in df.columns else pd.Series([True] * len(df))
    accepted_arr = accepted.to_numpy()
    xtick_labels = _iteration_tick_labels(df)

    metrics_to_plot = [
        ("avg logL/token assigned (higher better)", ["avg_logL_per_token_assigned", "avg_logL_per_token"]),
        ("PPL token assigned (lower better)", ["PPL_tok_assigned", "PPL_tok", "perplexity_token_assigned"]),
        ("AIC (lower better)", ["AIC"]),
        ("BIC (lower better)", ["BIC"]),
        ("# clusters", ["clusters"]),
        ("logL total assigned (higher better)", ["logL_total_assigned", "logL_total"]),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes_flat = axes.flatten()
    fig.suptitle(f"EM Metrics Over Time: {run.run_dir.name}", fontsize=14, fontweight="bold")

    for ax, (title, candidates) in zip(axes_flat, metrics_to_plot):
        col = _pick_column(df, candidates)
        if col is None:
            ax.text(0.5, 0.5, "Metric not available", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.grid(True, alpha=0.25)
            continue

        y = df[col].to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=2, label=col)
        if (~accepted_arr).any():
            ax.scatter(
                x[~accepted_arr],
                y[~accepted_arr],
                color="red",
                marker="x",
                s=70,
                linewidths=2,
                label="rejected iteration",
            )

        ax.set_title(title)
        ax.set_xlabel("iteration / operation")
        ax.set_ylabel(col)
        ax.set_xticks(x)
        ax.set_xticklabels(xtick_labels, rotation=35, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out_path = run.metrics_csv.parent / "em_metrics_over_time.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _make_summary_rows(runs: list[RunMetrics]) -> pd.DataFrame:
    rows = []
    for run in runs:
        df = run.df
        df_no_init = df[df["iter"] > 0].copy() if "iter" in df.columns else df.copy()
        accepted_no_init = (
            _parse_accepted(df_no_init["accepted"])
            if ("accepted" in df_no_init.columns and not df_no_init.empty)
            else pd.Series([], dtype=bool)
        )
        row = {
            "run_name": run.run_dir.name,
            "metrics_csv": str(run.metrics_csv),
            "n_points_in_plot": int(len(df)),
            "n_iters": int(len(df_no_init)),
            "final_iter": int(df["iter"].max()) if "iter" in df.columns else int(len(df_no_init)),
            "accepted_iters": int(accepted_no_init.sum()) if not accepted_no_init.empty else int(len(df_no_init)),
            "rejected_iters": int((~accepted_no_init).sum()) if not accepted_no_init.empty else 0,
        }

        clusters_col = _pick_column(df, ["clusters"])
        avg_col = _pick_column(df, ["avg_logL_per_token_assigned", "avg_logL_per_token"])
        ppl_col = _pick_column(df, ["PPL_tok_assigned", "PPL_tok", "perplexity_token_assigned"])
        logl_col = _pick_column(df, ["logL_total_assigned", "logL_total"])
        aic_col = _pick_column(df, ["AIC"])
        bic_col = _pick_column(df, ["BIC"])

        if clusters_col:
            row["final_clusters"] = float(df[clusters_col].iloc[-1])
        if avg_col:
            row["final_avg_logL_tok"] = float(df[avg_col].iloc[-1])
            row["best_avg_logL_tok"] = float(df[avg_col].max())
        if ppl_col:
            row["final_PPL_tok"] = float(df[ppl_col].iloc[-1])
            row["best_PPL_tok"] = float(df[ppl_col].min())
        if logl_col:
            row["final_logL_total"] = float(df[logl_col].iloc[-1])
            row["best_logL_total"] = float(df[logl_col].max())
        if aic_col:
            row["final_AIC"] = float(df[aic_col].iloc[-1])
            row["best_AIC"] = float(df[aic_col].min())
        if bic_col:
            row["final_BIC"] = float(df[bic_col].iloc[-1])
            row["best_BIC"] = float(df[bic_col].min())

        rows.append(row)
    return pd.DataFrame(rows)


def _plot_multi_run_comparison(runs: list[RunMetrics], output_dir: Path, dpi: int = 170) -> Path | None:
    if len(runs) < 2:
        return None

    metric_specs = [
        ("avg logL/token assigned", ["avg_logL_per_token_assigned", "avg_logL_per_token"], "higher better"),
        ("PPL token assigned", ["PPL_tok_assigned", "PPL_tok", "perplexity_token_assigned"], "lower better"),
        ("# clusters", ["clusters"], ""),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, (title, candidates, subtitle) in zip(axes, metric_specs):
        plotted_any = False
        for run in runs:
            col = _pick_column(run.df, candidates)
            if col is None:
                continue
            x = run.df["iter"].astype(int).to_numpy()
            y = run.df[col].to_numpy(dtype=float)
            label = f"{run.run_dir.name} (final={y[-1]:.4g})"
            ax.plot(x, y, marker="o", linewidth=1.8, label=label)
            plotted_any = True

        full_title = title if not subtitle else f"{title} ({subtitle})"
        ax.set_title(full_title)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
        if plotted_any:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No metric available", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(f"EM Run Comparison ({len(runs)} runs)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = output_dir / "em_runs_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot EM metrics over iterations from EM diagnostics CSVs.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help=(
            "Either a single run directory containing em_diagnostics/em_iteration_metrics_summary.csv, "
            "or a parent directory with multiple run folders."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory for multi-run outputs (comparison png + summary csv). Defaults to input_dir.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for em_diagnostics/em_iteration_metrics_summary.csv files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=170,
        help="Figure DPI for saved plots.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    metrics_csvs = _find_metrics_csvs(input_dir, recursive=args.recursive)
    if not metrics_csvs:
        raise FileNotFoundError(
            f"No em_iteration_metrics_summary.csv found under: {input_dir}\n"
            "Expected path pattern: <run_dir>/em_diagnostics/em_iteration_metrics_summary.csv"
        )

    runs = _load_runs(metrics_csvs)
    if not runs:
        raise RuntimeError("No valid non-empty metrics CSVs found.")

    print(f"[info] found {len(runs)} run(s).")

    per_run_outputs = []
    for run in runs:
        out = _plot_single_run(run, dpi=args.dpi)
        per_run_outputs.append(out)
        print(f"[ok] wrote per-run plot: {out}")

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(runs) > 1:
        comparison_path = _plot_multi_run_comparison(runs, output_dir=output_dir, dpi=args.dpi)
        if comparison_path is not None:
            print(f"[ok] wrote comparison plot: {comparison_path}")

        summary_df = _make_summary_rows(runs).sort_values("run_name").reset_index(drop=True)
        summary_path = output_dir / "em_runs_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"[ok] wrote summary table: {summary_path}")

    print("[done]")


if __name__ == "__main__":
    main()
