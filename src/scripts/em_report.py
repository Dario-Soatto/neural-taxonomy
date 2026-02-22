import argparse
import re
from pathlib import Path

import pandas as pd


def parse_em_log(log_path: Path):
    patterns = {
        "ELBO": re.compile(r"ELBO[:=]\s*([-\d\.]+)"),
        "AIC": re.compile(r"AIC[:=]\s*([-\d\.]+)"),
        "BIC": re.compile(r"BIC[:=]\s*([-\d\.]+)"),
        "L_baseline": re.compile(r"L_baseline[:=]\s*([-\d\.]+)"),
        "logL_total": re.compile(r"logL_total[:=]\s*([-\d\.]+)"),
    }
    remove_margin_pat = re.compile(r"Remove margin.*min/median/max:\s*([-\d\.]+)/([-\d\.]+)/([-\d\.]+)")
    split_margin_pat = re.compile(r"Split margin.*min/median/max:\s*([-\d\.]+)/([-\d\.]+)/([-\d\.]+)")
    schema_pattern = re.compile(r"(split|merge|remove|add|schema)", re.IGNORECASE)
    noise_pattern = re.compile(r"Adding requests|Processed prompts|FutureWarning", re.IGNORECASE)

    metrics = {k: [] for k in patterns}
    margins = {"remove": [], "split": []}
    schema_events = []

    if not log_path.exists():
        return metrics, schema_events

    with log_path.open() as f:
        for line in f:
            for key, pat in patterns.items():
                m = pat.search(line)
                if m:
                    try:
                        metrics[key].append(float(m.group(1)))
                    except ValueError:
                        pass
            m = remove_margin_pat.search(line)
            if m:
                margins["remove"].append(tuple(map(float, m.groups())))
            m = split_margin_pat.search(line)
            if m:
                margins["split"].append(tuple(map(float, m.groups())))
            if schema_pattern.search(line) and not noise_pattern.search(line):
                schema_events.append(line.rstrip())

    return metrics, schema_events, margins


def write_clean_log(log_path: Path, out_path: Path):
    # Remove progress bars and noisy vLLM/tqdm lines while keeping EM metrics.
    noisy = re.compile(
        r"(Adding requests|Processed prompts|Scoring sentences|Batches:|Generating prompts:|Loading safetensors|Capturing CUDA graphs|Labeling nodes|Clustering|FutureWarning)",
        re.IGNORECASE,
    )
    datapoint = re.compile(r"Datapoint \(idx", re.IGNORECASE)
    with log_path.open() as src, out_path.open("w") as dst:
        for line in src:
            if noisy.search(line) or datapoint.search(line):
                continue
            dst.write(line)


def summarize_scores(csv_path: Path):
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return None
    df["pmax"] = df[prob_cols].max(axis=1)
    cluster_counts = df["selected_agglomerative_cluster_label"].value_counts()
    return {
        "rows": len(df),
        "prob_cols": prob_cols,
        "pmax_desc": df["pmax"].describe(),
        "cluster_counts": cluster_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to EM log (e.g., logs/em_*.err)")
    ap.add_argument("--csv", required=True, help="Path to em_refined_scores CSV")
    ap.add_argument("--quiet_log", action="store_true", help="Print EM metrics only (hide progress bars & datapoint logs).")
    ap.add_argument("--write_clean_log", help="Write a filtered copy of the log without progress bars/datapoint lines.")
    args = ap.parse_args()

    log_path = Path(args.log)
    csv_path = Path(args.csv)

    metrics, schema_events, margins = parse_em_log(log_path)
    summary = summarize_scores(csv_path)

    print("EM report")
    print(f"- log: {log_path}")
    print(f"- csv: {csv_path}")

    if summary is None:
        print("- score summary: not available (missing file or prob columns)")
    else:
        print(f"- rows: {summary['rows']}")
        print(f"- prob columns: {summary['prob_cols']}")
        print("- pmax summary:")
        print(summary["pmax_desc"])
        print("- cluster counts:")
        print(summary["cluster_counts"])

    if any(len(v) > 0 for v in metrics.values()):
        print("- iteration metrics (in order found):")
        for key, vals in metrics.items():
            if vals:
                print(f"  {key}: {vals}")
    else:
        print("- iteration metrics: not found in log")

    if margins["remove"] or margins["split"]:
        print("- threshold margins (min/median/max):")
        if margins["remove"]:
            print(f"  remove: {margins['remove']}")
        if margins["split"]:
            print(f"  split: {margins['split']}")

    if args.write_clean_log:
        out_path = Path(args.write_clean_log)
        write_clean_log(log_path, out_path)
        print(f"- wrote cleaned log: {out_path}")

    if args.quiet_log:
        # Filter down to EM metrics lines only
        metrics_pattern = re.compile(r"(\\[E-step\\]|\\[Iter\\s+\\d+\\]|ELBO|AIC|BIC|L_baseline|logL_total|PPL|Remove margin|Split margin)", re.IGNORECASE)
        with log_path.open() as f:
            for line in f:
                if metrics_pattern.search(line):
                    print(line.rstrip())
        return

    if schema_events:
        print("- schema events:")
        for line in schema_events:
            print(f"  {line}")
    else:
        print("- schema events: none found")


if __name__ == "__main__":
    main()
