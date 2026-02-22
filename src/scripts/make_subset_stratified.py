#!/usr/bin/env python3
import argparse
import csv
import os
import random


def build_doc_counts(path, doc_idx):
    counts = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            doc = row[doc_idx]
            counts[doc] = counts.get(doc, 0) + 1
    return counts


def stratified_sample(path, doc_idx, per_doc_map, seed):
    random.seed(seed)
    reservoirs = {doc: [] for doc, k in per_doc_map.items() if k > 0}
    seen = {doc: 0 for doc, k in per_doc_map.items() if k > 0}

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            doc = row[doc_idx]
            k = per_doc_map.get(doc, 0)
            if k <= 0:
                continue
            seen[doc] += 1
            bucket = reservoirs[doc]
            if len(bucket) < k:
                bucket.append(row)
            else:
                j = random.randint(1, seen[doc])
                if j <= k:
                    bucket[j - 1] = row
    return reservoirs


def fill_remaining(path, doc_idx, selected_ids, remaining, seed):
    random.seed(seed)
    if remaining <= 0:
        return []
    reservoir = []
    seen = 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            doc = row[doc_idx]
            sent = row[doc_idx + 1] if doc_idx + 1 < len(row) else ""
            key = (doc, sent, row[doc_idx + 2] if doc_idx + 2 < len(row) else "")
            if key in selected_ids:
                continue
            seen += 1
            if len(reservoir) < remaining:
                reservoir.append(row)
            else:
                j = random.randint(1, seen)
                if j <= remaining:
                    reservoir[j - 1] = row
    return reservoir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="processed_custom_input.csv", help="Input CSV")
    ap.add_argument("--output", default="experiments/wiki_biographies/data/processed_custom_input_subset_10000.csv", help="Output CSV")
    ap.add_argument("--n", type=int, default=10000, help="Target number of rows")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"Input not found: {args.input}")

    # Find doc_index column index.
    with open(args.input, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if "doc_index" not in header:
        raise SystemExit("Input CSV must contain 'doc_index' column.")
    doc_idx = header.index("doc_index")

    counts = build_doc_counts(args.input, doc_idx)
    docs = list(counts.keys())
    num_docs = len(docs)

    # Determine per-doc targets.
    if num_docs >= args.n:
        random.seed(args.seed)
        chosen_docs = set(random.sample(docs, args.n))
        per_doc_map = {doc: (1 if doc in chosen_docs else 0) for doc in docs}
    else:
        base = args.n // num_docs
        remainder = args.n - base * num_docs
        random.seed(args.seed)
        extra_docs = set(random.sample(docs, remainder)) if remainder > 0 else set()
        per_doc_map = {doc: base + (1 if doc in extra_docs else 0) for doc in docs}

    reservoirs = stratified_sample(args.input, doc_idx, per_doc_map, args.seed)
    rows = []
    selected_ids = set()
    for doc, bucket in reservoirs.items():
        for row in bucket:
            sent = row[doc_idx + 1] if doc_idx + 1 < len(row) else ""
            key = (doc, sent, row[doc_idx + 2] if doc_idx + 2 < len(row) else "")
            if key not in selected_ids:
                selected_ids.add(key)
                rows.append(row)

    # If some docs had fewer rows than requested, fill remaining randomly.
    if len(rows) < args.n:
        fill = fill_remaining(args.input, doc_idx, selected_ids, args.n - len(rows), args.seed + 1)
        rows.extend(fill)

    # Final trim in case we overshot.
    if len(rows) > args.n:
        random.seed(args.seed)
        rows = random.sample(rows, args.n)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
