import argparse
import glob
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


def summarize_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    df = df.sort_values("iter").reset_index(drop=True)
    final = df.iloc[-1]
    score_col = "avg_logL_per_token" if "avg_logL_per_token" in df.columns else "logL_total"
    best = df.loc[df[score_col].idxmax()]
    accepted_iters = int(df["accepted"].sum()) if "accepted" in df.columns else None
    out = {
        "n_iters": int(len(df)),
        "accepted_iters": accepted_iters,
        "best_iter": int(best["iter"]),
        "best_logL_total": float(best["logL_total"]) if "logL_total" in best else None,
        "best_avg_logL_per_token": float(best["avg_logL_per_token"]) if "avg_logL_per_token" in best else None,
        "final_iter": int(final["iter"]),
        "final_logL_total": float(final["logL_total"]) if "logL_total" in final else None,
        "final_avg_logL_per_token": float(final["avg_logL_per_token"]) if "avg_logL_per_token" in final else None,
        "final_PPL_tok": float(final["PPL_tok"]) if "PPL_tok" in final else None,
        "final_mismatch_rate": float(final["mismatch_rate"]) if "mismatch_rate" in final and pd.notna(final["mismatch_rate"]) else None,
        "final_pmax_median": float(final["pmax_median"]) if "pmax_median" in final and pd.notna(final["pmax_median"]) else None,
        "final_entropy_median": float(final["entropy_median"]) if "entropy_median" in final and pd.notna(final["entropy_median"]) else None,
    }
    return out


def collect_runs(grid_root: Path, operation: str, profile: str) -> pd.DataFrame:
    op_dir = grid_root / f"{operation}_{profile}"
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(op_dir.glob(f"{operation}_{profile}_*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "em_diagnostics" / "em_iteration_metrics_summary.csv"
        row: dict[str, Any] = {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "has_metrics": False,
        }
        summary = summarize_metrics(metrics_path)
        if summary is not None:
            row["has_metrics"] = True
            row.update(summary)
        rows.append(row)
    return pd.DataFrame(rows)


def parse_params_from_logs(log_files: list[Path]) -> dict[str, str]:
    # Example line:
    # [4/18] revise_coarse_004: {"--revise_confidence_max": 0.3, ...}
    pat = re.compile(r"\[\d+/\d+\]\s+([A-Za-z0-9_]+):\s+(\{.*\})")
    out: dict[str, str] = {}
    for p in log_files:
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines():
            m = pat.search(line)
            if not m:
                continue
            run_name = m.group(1).strip()
            payload = m.group(2).strip()
            try:
                obj = json.loads(payload)
                out[run_name] = json.dumps(obj, sort_keys=True)
            except Exception:
                out[run_name] = payload
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot one EM grid operation: map run->params and summarize available metrics."
    )
    parser.add_argument("--grid_root", required=True, type=str)
    parser.add_argument("--operation", required=True, type=str)
    parser.add_argument("--profile", default="coarse", type=str)
    parser.add_argument("--logs_glob", default="logs/em_*.out", type=str)
    parser.add_argument("--output_dir", default="logs", type=str)
    parser.add_argument("--top_k", default=10, type=int)
    args = parser.parse_args()

    grid_root = Path(args.grid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_runs(grid_root, args.operation, args.profile)
    if df.empty:
        raise SystemExit(f"No run directories found for {args.operation}_{args.profile} under {grid_root}")

    log_files = [Path(p) for p in sorted(glob.glob(args.logs_glob))]
    params_map = parse_params_from_logs(log_files)
    df["params_json"] = df["run_name"].map(params_map).fillna("")

    param_csv = output_dir / f"{args.operation}_{args.profile}_param_map.csv"
    metrics_csv = output_dir / f"{args.operation}_{args.profile}_metrics_snapshot.csv"
    report_txt = output_dir / f"{args.operation}_{args.profile}_analysis.txt"

    param_df = df[["run_name", "params_json"]].copy()
    param_df.to_csv(param_csv, index=False)
    df.to_csv(metrics_csv, index=False)

    lines: list[str] = []
    lines.append(f"=== {args.operation.upper()} {args.profile} Snapshot ===")
    lines.append(f"grid_root={grid_root}")
    lines.append(f"runs_total={len(df)}")
    lines.append(f"runs_with_metrics={int(df['has_metrics'].sum())}")
    lines.append("")
    lines.append(f"param_csv={param_csv}")
    lines.append(f"metrics_csv={metrics_csv}")
    lines.append("")

    done = df[df["has_metrics"] == True].copy()
    if done.empty:
        lines.append("No completed metrics yet.")
    else:
        score_col = "best_avg_logL_per_token" if "best_avg_logL_per_token" in done.columns else "best_logL_total"
        top = done.sort_values(score_col, ascending=False).head(args.top_k)
        keep = [
            "run_name",
            "best_iter",
            "best_avg_logL_per_token",
            "best_logL_total",
            "final_avg_logL_per_token",
            "final_PPL_tok",
            "accepted_iters",
            "n_iters",
            "params_json",
        ]
        keep = [c for c in keep if c in top.columns]
        lines.append(f"Top {min(args.top_k, len(top))} runs by {score_col}:")
        lines.append(top[keep].to_string(index=False))

    report_txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote {param_csv}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {report_txt}")


if __name__ == "__main__":
    main()
