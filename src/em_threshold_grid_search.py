import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


OPERATION_GRIDS = {
    "split": {
        "coarse": {
            "--split_ll_margin": [4.0, 6.0, 8.0],
            "--split_confidence_max": [0.20, 0.30, 0.40],
            "--split_gap_median_min": [0.08, 0.12, 0.18],
            "--split_min_cluster_size": [15, 25],
            "--split_min_conditions": [2],
        },
        "fast": {
            "--split_ll_margin": [5.0, 7.0],
            "--split_confidence_max": [0.25, 0.35],
            "--split_gap_median_min": [0.10, 0.16],
            "--split_min_conditions": [2],
        },
        # Non-fast structural tuning profile (exactly 10 configs):
        # 5 (confidence) x 2 (ll margin) = 10
        "structural": {
            "--split_ll_margin": [4.0, 7.0],
            "--split_confidence_max": [0.14, 0.20, 0.26, 0.32, 0.38],
            "--split_gap_median_min": [0.16],
            "--split_min_cluster_size": [25],
            "--split_min_conditions": [2],
        },
    },
    "merge": {
        "coarse": {
            "--merge_similarity_min": [0.78, 0.84, 0.90],
            "--merge_l_diff_ratio_max": [0.08, 0.15],
            "--merge_c_diff_ratio_max": [0.08, 0.15],
            "--merge_min_conditions": [2],
        },
        "fast": {
            "--merge_similarity_min": [0.80, 0.88],
            "--merge_l_diff_ratio_max": [0.10],
            "--merge_c_diff_ratio_max": [0.10],
            "--merge_min_conditions": [2],
        },
        # Non-fast structural tuning profile (exactly 10 configs):
        # 5 (similarity) x 2 (L-diff ratio) = 10
        "structural": {
            "--merge_similarity_min": [0.55, 0.65, 0.75, 0.85, 0.92],
            "--merge_l_diff_ratio_max": [0.06, 0.12],
            "--merge_c_diff_ratio_max": [0.15],
            "--merge_min_conditions": [2],
        },
    },
    "remove": {
        "coarse": {
            "--remove_min_cluster_size": [3, 5, 8],
            "--remove_ll_factor": [1.00, 1.01, 1.03],
        },
        "fast": {
            "--remove_min_cluster_size": [4, 7],
            "--remove_ll_factor": [1.00, 1.02],
        },
        # Non-fast structural tuning profile (exactly 10 configs):
        # 2 (size) x 5 (ll_factor) = 10
        # Includes looser ll_factor values to expose remove behavior in tiny, weak clusters.
        "structural": {
            "--remove_min_cluster_size": [4, 8],
            "--remove_ll_factor": [0.55, 0.65, 0.75, 0.85, 0.95],
        },
    },
    "revise": {
        "coarse": {
            "--revise_ll_margin": [3.0, 4.0, 6.0],
            "--revise_confidence_max": [0.20, 0.30, 0.40],
            "--revise_min_conditions": [2],
        },
        "fast": {
            "--revise_ll_margin": [3.0, 5.0],
            "--revise_confidence_max": [0.25, 0.35],
            "--revise_min_conditions": [2],
        },
        # Non-fast structural tuning profile (exactly 10 configs):
        # 5 (ll margin) x 2 (confidence) = 10
        "structural": {
            "--revise_ll_margin": [2.0, 3.5, 5.0, 6.5, 8.0],
            "--revise_confidence_max": [0.15, 0.30],
            "--revise_min_conditions": [2],
        },
    },
    "add": {
        "coarse": {
            "--add_low_confidence_max": [0.15, 0.20, 0.25],
            "--add_min_poorly_explained": [10, 25, 50],
            "--add_items_per_new_cluster": [40, 80, 120],
        },
        "fast": {
            "--add_low_confidence_max": [0.18, 0.24],
            "--add_min_poorly_explained": [15, 35],
            "--add_items_per_new_cluster": [50, 100],
        },
        # Non-fast structural tuning profile (exactly 10 configs):
        # 5 (low_confidence_max) x 2 (dynamic quantile) = 10
        # Looser grouping/diversity gates to expose add behavior.
        "structural": {
            "--add_low_confidence_max": [0.18, 0.24, 0.30, 0.36, 0.42],
            "--add_low_confidence_quantile": [0.05, 0.15],
            "--add_min_poorly_explained": [8],
            "--add_items_per_new_cluster": [20],
            "--add_min_group_size": [2],
            "--add_entropy_min": [0.75],
            "--add_cohesion_min": [0.05],
        },
    },
}


def build_param_combinations(grid: dict[str, list]) -> list[dict[str, object]]:
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combos.append(dict(zip(keys, values)))
    return combos


def normalize_base_args(base_args: list[str]) -> list[str]:
    if base_args and base_args[0] == "--":
        return base_args[1:]
    return base_args


def build_command(
    python_exec: str,
    run_script_path: Path,
    base_args: list[str],
    operation: str,
    combo: dict[str, object],
    output_file: Path,
    diagnostics_dir: Path,
) -> list[str]:
    cmd = [
        python_exec,
        "-u",
        str(run_script_path),
        *base_args,
        "--tune_operation",
        operation,
        f"--{operation}_enabled",
        "--output_file",
        str(output_file),
        "--diagnostics_dir",
        str(diagnostics_dir),
        "--log_iteration_metrics",
    ]
    for k, v in combo.items():
        cmd.extend([k, str(v)])
    return cmd


def run_with_live_log(
    cmd: list[str],
    log_path: Path,
) -> tuple[int, list[str]]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    tail_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                print(line, end="", flush=True)
                tail_lines.append(line.rstrip("\n"))
                if len(tail_lines) > 80:
                    tail_lines = tail_lines[-80:]
        return_code = proc.wait()
    return return_code, tail_lines[-20:]


def summarize_run(diag_dir: Path) -> dict[str, object]:
    metrics_csv = diag_dir / "em_iteration_metrics_summary.csv"
    if not metrics_csv.exists():
        return {"has_metrics": False}

    df = pd.read_csv(metrics_csv)
    if df.empty:
        return {"has_metrics": False}

    df = df.sort_values("iter")
    final_row = df.iloc[-1]
    best_logl_row = df.loc[df["logL_total"].idxmax()]
    best_tokll_row = df.loc[df["avg_logL_per_token"].idxmax()] if "avg_logL_per_token" in df.columns else None

    out = {
        "has_metrics": True,
        "final_iter": int(final_row["iter"]),
        "final_clusters": int(final_row["clusters"]),
        "final_logL_total": float(final_row["logL_total"]),
        "final_AIC": float(final_row["AIC"]),
        "final_BIC": float(final_row["BIC"]),
        "final_PPL": float(final_row["PPL"]),
        "final_PPL_tok": float(final_row["PPL_tok"]) if "PPL_tok" in final_row else None,
        "final_avg_logL_per_token": float(final_row["avg_logL_per_token"]) if "avg_logL_per_token" in final_row else None,
        "final_ELBO": float(final_row["ELBO"]),
        "final_mismatch_rate": float(final_row["mismatch_rate"]) if pd.notna(final_row["mismatch_rate"]) else None,
        "final_pmax_median": float(final_row["pmax_median"]) if pd.notna(final_row["pmax_median"]) else None,
        "final_entropy_median": float(final_row["entropy_median"]) if pd.notna(final_row["entropy_median"]) else None,
        "best_logL_iter": int(best_logl_row["iter"]),
        "best_logL_total": float(best_logl_row["logL_total"]),
    }
    if best_tokll_row is not None:
        out["best_avg_logL_tok_iter"] = int(best_tokll_row["iter"])
        out["best_avg_logL_per_token"] = float(best_tokll_row["avg_logL_per_token"])
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Grid-search EM thresholds for one operation at a time."
    )
    parser.add_argument("--operation", choices=["split", "merge", "remove", "add", "revise"], required=True)
    parser.add_argument("--grid_profile", choices=["coarse", "fast", "structural"], default="coarse")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument(
        "base_run_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_em_algorithm.py. Pass after '--'.",
    )
    args = parser.parse_args()

    base_args = normalize_base_args(args.base_run_args)
    if not base_args:
        raise ValueError("No base_run_args provided. Pass run_em_algorithm args after '--'.")

    run_script_path = Path(__file__).resolve().parent / "run_em_algorithm.py"
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    grid = OPERATION_GRIDS[args.operation][args.grid_profile]
    combos = build_param_combinations(grid)
    print(f"Running {len(combos)} combinations for operation='{args.operation}' profile='{args.grid_profile}'.", flush=True)

    records = []
    for run_idx, combo in enumerate(combos, start=1):
        run_name = f"{args.operation}_{args.grid_profile}_{run_idx:03d}"
        run_dir = output_root / run_name
        diag_dir = run_dir / "em_diagnostics"
        output_file = run_dir / "em_refined_scores.csv"
        run_log_path = run_dir / "run.log"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_command(
            python_exec=args.python_executable,
            run_script_path=run_script_path,
            base_args=base_args,
            operation=args.operation,
            combo=combo,
            output_file=output_file,
            diagnostics_dir=diag_dir,
        )
        print(f"[{run_idx}/{len(combos)}] {run_name}: {json.dumps(combo, sort_keys=True)}", flush=True)
        print(f"  -> writing live log to {run_log_path}", flush=True)
        return_code, tail_lines = run_with_live_log(cmd, run_log_path)

        record = {
            "run_name": run_name,
            "operation": args.operation,
            "grid_profile": args.grid_profile,
            "return_code": return_code,
            "params_json": json.dumps(combo, sort_keys=True),
            "run_dir": str(run_dir),
            "diagnostics_dir": str(diag_dir),
            "output_file": str(output_file),
            "run_log_file": str(run_log_path),
        }
        for k, v in combo.items():
            col_name = k.lstrip("-").replace("-", "_")
            record[col_name] = v
        if return_code != 0:
            record["error_tail"] = "\n".join(tail_lines)
            records.append(record)
            continue

        summary = summarize_run(diag_dir)
        record.update(summary)
        records.append(record)

    results_df = pd.DataFrame(records)
    summary_path = output_root / f"grid_search_{args.operation}_{args.grid_profile}_summary.csv"
    results_df.to_csv(summary_path, index=False)
    print(f"Wrote grid-search summary: {summary_path}", flush=True)

    successful = results_df[(results_df["return_code"] == 0) & (results_df.get("has_metrics") == True)]
    if not successful.empty:
        top = successful.sort_values("best_logL_total", ascending=False).head(10)
        print("Top runs by best_logL_total:", flush=True)
        cols = [
            "run_name",
            "best_logL_total",
            "best_avg_logL_per_token",
            "best_logL_iter",
            "best_avg_logL_tok_iter",
            "final_logL_total",
            "final_avg_logL_per_token",
            "final_AIC",
            "final_BIC",
            "final_PPL_tok",
            "final_mismatch_rate",
            "final_pmax_median",
            "params_json",
        ]
        cols = [c for c in cols if c in top.columns]
        print(top[cols].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
