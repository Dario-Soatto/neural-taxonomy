import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


OPERATIONS = ("split", "merge", "remove", "add", "revise")


def parse_params(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def summarize_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None
    df = df.sort_values("iter").reset_index(drop=True)
    final = df.iloc[-1]
    if "avg_logL_per_token" in df.columns:
        best_idx = df["avg_logL_per_token"].idxmax()
    else:
        best_idx = df["logL_total"].idxmax()
    best = df.loc[best_idx]
    out = {
        "metrics_path": str(metrics_path),
        "n_iters": int(len(df)),
        "final_iter": int(final["iter"]),
        "final_clusters": int(final["clusters"]) if "clusters" in final else None,
        "final_logL_total": float(final["logL_total"]) if "logL_total" in final else None,
        "final_avg_logL_per_token": float(final["avg_logL_per_token"]) if "avg_logL_per_token" in final else None,
        "final_PPL_tok": float(final["PPL_tok"]) if "PPL_tok" in final else None,
        "final_mismatch_rate": float(final["mismatch_rate"]) if "mismatch_rate" in final and pd.notna(final["mismatch_rate"]) else None,
        "final_pmax_median": float(final["pmax_median"]) if "pmax_median" in final and pd.notna(final["pmax_median"]) else None,
        "final_entropy_median": float(final["entropy_median"]) if "entropy_median" in final and pd.notna(final["entropy_median"]) else None,
        "best_iter": int(best["iter"]),
        "best_logL_total": float(best["logL_total"]) if "logL_total" in best else None,
        "best_avg_logL_per_token": float(best["avg_logL_per_token"]) if "avg_logL_per_token" in best else None,
    }
    return out


def collect_operation_rows(grid_root: Path, operation: str, profile: str) -> pd.DataFrame:
    op_dir = grid_root / f"{operation}_{profile}"
    summary_csv = op_dir / f"grid_search_{operation}_{profile}_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        if "run_dir" in df.columns:
            for idx, row in df.iterrows():
                run_dir = Path(str(row.get("run_dir", "")))
                metrics_path = run_dir / "em_diagnostics" / "em_iteration_metrics_summary.csv"
                if metrics_path.exists():
                    df.loc[idx, "metrics_path"] = str(metrics_path)
        df["source"] = "summary_csv"
        return df

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(op_dir.glob(f"{operation}_{profile}_*")):
        if not run_dir.is_dir():
            continue
        metrics_path = run_dir / "em_diagnostics" / "em_iteration_metrics_summary.csv"
        rec: dict[str, Any] = {
            "run_name": run_dir.name,
            "operation": operation,
            "grid_profile": profile,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "return_code": None,
            "has_metrics": False,
            "params_json": "",
            "source": "run_dirs_partial",
        }
        m = summarize_metrics(metrics_path)
        if m is not None:
            rec.update(m)
            rec["has_metrics"] = True
        rows.append(rec)
    return pd.DataFrame(rows)


def load_best_run_metrics(row: pd.Series) -> pd.DataFrame | None:
    metrics_path = row.get("metrics_path")
    if metrics_path is None or str(metrics_path).strip() == "":
        return None
    p = Path(str(metrics_path))
    if not p.exists():
        return None
    df = pd.read_csv(p).sort_values("iter").reset_index(drop=True)
    return df if not df.empty else None


def trend_assessment(df: pd.DataFrame) -> dict[str, Any]:
    first = df.iloc[0]
    last = df.iloc[-1]
    out: dict[str, Any] = {}
    if "avg_logL_per_token" in df.columns:
        out["delta_avg_logL_per_token"] = float(last["avg_logL_per_token"] - first["avg_logL_per_token"])
    if "PPL_tok" in df.columns:
        out["delta_PPL_tok"] = float(last["PPL_tok"] - first["PPL_tok"])
    if "mismatch_rate" in df.columns and pd.notna(first.get("mismatch_rate")) and pd.notna(last.get("mismatch_rate")):
        out["delta_mismatch_rate"] = float(last["mismatch_rate"] - first["mismatch_rate"])
    if "pmax_median" in df.columns and pd.notna(first.get("pmax_median")) and pd.notna(last.get("pmax_median")):
        out["delta_pmax_median"] = float(last["pmax_median"] - first["pmax_median"])
    if "entropy_median" in df.columns and pd.notna(first.get("entropy_median")) and pd.notna(last.get("entropy_median")):
        out["delta_entropy_median"] = float(last["entropy_median"] - first["entropy_median"])
    return out


def format_float(x: Any, nd: int = 6) -> str:
    if x is None:
        return "NA"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def print_log_tails(log_files: list[Path], tail_lines: int) -> None:
    print("\n=== SLURM Log Tails ===")
    for p in log_files:
        print(f"\n--- {p} (last {tail_lines} lines) ---")
        if not p.exists():
            print("missing")
            continue
        lines = p.read_text(errors="replace").splitlines()
        for line in lines[-tail_lines:]:
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize EM grid tuning results across operations.")
    parser.add_argument("--grid_root", required=True, type=str, help="Root directory containing *_coarse grid folders.")
    parser.add_argument("--profile", default="coarse", type=str, help="Grid profile suffix (e.g., coarse, fast).")
    parser.add_argument("--tail_lines", default=80, type=int)
    parser.add_argument("--log-file", action="append", default=[], help="Optional SLURM log files to tail.")
    parser.add_argument("--output_file", default="", type=str, help="Optional path to save report text.")
    args = parser.parse_args()

    grid_root = Path(args.grid_root)
    sections: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        sections.append(line)

    emit("=== EM Grid Tuning Report ===")
    emit(f"grid_root={grid_root}")
    emit(f"profile={args.profile}")

    for op in OPERATIONS:
        emit()
        emit(f"=== {op.upper()} ===")
        df = collect_operation_rows(grid_root, op, args.profile)
        if df.empty:
            emit("no run directories or summary found")
            continue

        n_rows = len(df)
        n_done = int((df["return_code"] == 0).sum()) if "return_code" in df.columns else 0
        n_metrics = int((df["has_metrics"] == True).sum()) if "has_metrics" in df.columns else 0
        emit(f"rows={n_rows} completed={n_done} with_metrics={n_metrics} source={df['source'].iloc[0] if 'source' in df.columns else 'unknown'}")

        done = df[df["has_metrics"] == True].copy() if "has_metrics" in df.columns else pd.DataFrame()
        if done.empty:
            emit("no completed metric rows yet")
            continue

        score_col = "best_avg_logL_per_token" if "best_avg_logL_per_token" in done.columns else "best_logL_total"
        top = done.sort_values(score_col, ascending=False).head(5)
        cols = [
            "run_name",
            score_col,
            "best_logL_total",
            "final_avg_logL_per_token",
            "final_PPL_tok",
            "final_mismatch_rate",
            "final_pmax_median",
            "final_entropy_median",
            "params_json",
        ]
        cols = [c for c in cols if c in top.columns]
        emit(top[cols].to_string(index=False))

        best_row = top.iloc[0]
        params = parse_params(best_row.get("params_json"))
        emit(f"\nbest_run={best_row.get('run_name','NA')}")
        emit(f"best_params={json.dumps(params, sort_keys=True) if params else 'NA (summary pending or unavailable)'}")

        best_metrics = load_best_run_metrics(best_row)
        if best_metrics is None:
            emit("best-run iteration metrics file missing")
            continue
        keep = ["iter", "clusters", "splits", "merges", "removes", "adds", "revises", "accepted", "logL_total", "avg_logL_per_token", "PPL_tok", "mismatch_rate", "pmax_median", "entropy_median"]
        keep = [c for c in keep if c in best_metrics.columns]
        emit("\niteration_trend:")
        emit(best_metrics[keep].to_string(index=False))

        deltas = trend_assessment(best_metrics)
        emit("\ndelta_first_to_last:")
        for k, v in deltas.items():
            emit(f"  {k}={format_float(v)}")

    if args.log_file:
        print_log_tails([Path(p) for p in args.log_file], args.tail_lines)

    if args.output_file:
        out = Path(args.output_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(sections) + "\n")
        print(f"\nSaved report to {out}")


if __name__ == "__main__":
    main()
