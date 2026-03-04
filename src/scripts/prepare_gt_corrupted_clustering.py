"""
Prepares hierarchy_results/ files for a near-ground-truth HuffPost clustering
with deliberate corruptions, as a sanity test for the EM algorithm.

Starting point: ground-truth label assignment
  - Each centroid is assigned to its plurality true_label category
  - Only the top --top_n_categories categories (by centroid count) are kept

Corruptions introduced (all chosen from well-represented categories):
  MERGES (2): two semantically distinct categories collapsed into one cluster
              → EM should detect incoherence and propose a SPLIT
  SPLITS (2): one category's centroids randomly divided into two sub-clusters
              with plausible-but-distinct labels
              → EM should detect similarity and propose a MERGE
  MISLABELS (2): correct document assignment, wrong cluster label
              → EM should detect mismatch and propose a REVISE

Usage:
    python src/scripts/prepare_gt_corrupted_clustering.py \
        --data_csv /path/to/huffpost_data.csv \
        --output_dir /path/to/hierarchy_results \
        --top_n_categories 15 \
        --seed 42
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# ── Deliberate corruptions ────────────────────────────────────────────────────
# Each merge collapses cat_B into cat_A (documents from both → same cluster id,
# label is kept as cat_A so it looks like a single-topic cluster).
MERGES = [
    ("POLITICS",  "FOOD & DRINK"),   # domestic politics vs food recipes
    ("WELLNESS",  "SPORTS"),         # personal health vs competitive sports
]

# Each split divides cat's centroids into two halves with different-sounding labels.
# The sub-label pairs sound distinct but all docs are really from the same category.
SPLITS = [
    ("ENTERTAINMENT", "Celebrity Culture",    "Film & Television"),
    ("PARENTING",     "Early Childhood Care", "Teen & Family Advice"),
]

# Each mislabel keeps the correct document assignment but uses a wrong label name.
MISLABELS = [
    ("BUSINESS",  "HOME IMPROVEMENT TIPS"),
    ("WEDDINGS",  "TECH STARTUPS"),
]
# ─────────────────────────────────────────────────────────────────────────────


def plurality_label(labels):
    """Return the most common label in a collection."""
    if len(labels) == 0:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True,
                        help="Path to huffpost_data.csv (output of prepare_huffpost_data.py).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the three hierarchy files into.")
    parser.add_argument("--top_n_categories", type=int, default=15,
                        help="Keep only the top-N categories by centroid count (default: 15).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    df = pd.read_csv(args.data_csv)
    logging.info("Loaded %d rows from %s.", len(df), args.data_csv)
    assert "centroid_id" in df.columns and "true_label" in df.columns, \
        "CSV must have 'centroid_id' and 'true_label' columns."

    # ── 2. Assign each centroid to its plurality true_label ───────────────────
    centroid_to_label = (
        df.groupby("centroid_id")["true_label"]
        .agg(plurality_label)
        .to_dict()
    )
    all_centroids = sorted(centroid_to_label.keys())
    logging.info("Found %d unique centroids.", len(all_centroids))

    # ── 3. Build initial cluster map: category → list of centroid_ids ─────────
    cat_to_centroids: dict[str, list[int]] = {}
    for cid in all_centroids:
        cat = centroid_to_label[cid]
        cat_to_centroids.setdefault(cat, []).append(cid)

    all_cats = sorted(cat_to_centroids.keys())
    logging.info("Found %d unique categories.", len(all_cats))
    for cat in all_cats:
        logging.info("  %-30s  %d centroids", cat, len(cat_to_centroids[cat]))

    # ── 3b. Filter to top-N categories ───────────────────────────────────────
    if args.top_n_categories > 0:
        top_cats = sorted(cat_to_centroids, key=lambda c: len(cat_to_centroids[c]), reverse=True)
        top_cats = top_cats[:args.top_n_categories]
        cat_to_centroids = {c: cat_to_centroids[c] for c in top_cats}
        logging.info("Filtered to top-%d categories: %s",
                     args.top_n_categories, sorted(top_cats))

    # ── 4. Apply MERGES ───────────────────────────────────────────────────────
    # cat_B centroids absorbed into cat_A; cat_B is removed from cat_to_centroids
    for cat_a, cat_b in MERGES:
        if cat_a not in cat_to_centroids:
            logging.warning("MERGE: cat_A '%s' not found, skipping.", cat_a)
            continue
        if cat_b not in cat_to_centroids:
            logging.warning("MERGE: cat_B '%s' not found, skipping.", cat_b)
            continue
        n_a = len(cat_to_centroids[cat_a])
        n_b = len(cat_to_centroids[cat_b])
        cat_to_centroids[cat_a].extend(cat_to_centroids.pop(cat_b))
        logging.info("MERGE: '%s' (%d) + '%s' (%d) → '%s' (%d centroids)",
                     cat_a, n_a, cat_b, n_b, cat_a,
                     len(cat_to_centroids[cat_a]))

    # ── 5. Apply SPLITS ───────────────────────────────────────────────────────
    # Randomly divide cat's centroids into two halves with new label names.
    for cat, label_a, label_b in SPLITS:
        if cat not in cat_to_centroids:
            logging.warning("SPLIT: cat '%s' not found, skipping.", cat)
            continue
        centroids = cat_to_centroids.pop(cat)
        if len(centroids) < 2:
            logging.warning("SPLIT: cat '%s' has only %d centroid(s); need ≥2, skipping.", cat, len(centroids))
            cat_to_centroids[cat] = centroids  # put it back unchanged
            continue
        shuffled = list(rng.permutation(centroids))
        half = len(shuffled) // 2
        cat_to_centroids[label_a] = shuffled[:half]
        cat_to_centroids[label_b] = shuffled[half:]
        logging.info("SPLIT: '%s' (%d) → '%s' (%d) + '%s' (%d)",
                     cat, len(shuffled),
                     label_a, len(shuffled[:half]),
                     label_b, len(shuffled[half:]))

    # ── 6. Apply MISLABELS ────────────────────────────────────────────────────
    # Rename the cluster label only (centroid assignments stay the same).
    mislabel_map = {}
    for true_cat, fake_label in MISLABELS:
        if true_cat not in cat_to_centroids:
            logging.warning("MISLABEL: cat '%s' not found, skipping.", true_cat)
            continue
        mislabel_map[true_cat] = fake_label
        logging.info("MISLABEL: '%s' will appear as '%s'", true_cat, fake_label)

    # ── 7. Build final cluster list with integer IDs ──────────────────────────
    final_cats = sorted(cat_to_centroids.keys())
    n_clusters = len(final_cats)
    cat_to_node_id = {cat: i for i, cat in enumerate(final_cats)}

    logging.info("Final schema: %d clusters.", n_clusters)

    # ── 8. Write clusters_level_1_N_clusters.csv ─────────────────────────────
    rows = []
    for cat, centroids in cat_to_centroids.items():
        node_id = cat_to_node_id[cat]
        for cid in centroids:
            rows.append({
                "centroid_id":    cid,
                "fcluster_label": node_id + 1,   # 1-indexed (matches random script)
                "graph_node_id":  node_id,
            })
    assign_df = pd.DataFrame(rows).sort_values("centroid_id").reset_index(drop=True)
    assign_path = out_dir / f"clusters_level_1_{n_clusters}_clusters.csv"
    assign_df.to_csv(assign_path, index=False)
    logging.info("Wrote %d centroid rows to %s.", len(assign_df), assign_path)

    # ── 9. Write inner_node_labels.csv ────────────────────────────────────────
    label_rows = []
    for cat in final_cats:
        node_id = cat_to_node_id[cat]
        display_label = mislabel_map.get(cat, cat)   # apply mislabel if any
        label_rows.append({
            "node_id":     node_id,
            "label":       display_label,
            "description": f"Cluster for {display_label}",
        })
    labels_df = pd.DataFrame(label_rows)
    labels_path = out_dir / "inner_node_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    logging.info("Wrote %d label rows to %s.", len(labels_df), labels_path)

    # ── 10. Write optimal_thresholds.csv ──────────────────────────────────────
    pd.DataFrame({
        "threshold":        [1.0],
        "n_clusters":       [n_clusters],
        "silhouette_score": [0.0],
    }).to_csv(out_dir / "optimal_thresholds.csv", index=False)
    logging.info("Wrote optimal_thresholds.csv with n_clusters=%d.", n_clusters)

    # ── 11. Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Ground-truth corrupted schema: {n_clusters} clusters")
    print("=" * 60)
    print(f"  Top-N categories:  {args.top_n_categories}")
    print(f"  Merges applied:    {len(MERGES)}")
    print(f"  Splits applied:    {len(SPLITS)}")
    print(f"  Mislabels applied: {len(MISLABELS)}")
    print("=" * 60)
    print(f"N_CLUSTERS={n_clusters}")   # last line, parseable by shell


if __name__ == "__main__":
    main()