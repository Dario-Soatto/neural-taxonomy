"""
Build ground-truth Newsgroups hierarchy variants for EM sanity checks.

Input:
  - CSV from prepare_newsgroups_data.py with columns:
      centroid_id, description, true_label

Output under --output_root:
  - hierarchy_results__clean
  - hierarchy_results__merge_only
  - hierarchy_results__split_only
  - hierarchy_results__remove_only
  - hierarchy_results__add_only

Each hierarchy directory contains:
  - clusters_level_1_<N>_clusters.csv
  - inner_node_labels.csv
  - optimal_thresholds.csv
  - corruption_manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def plurality_label(labels: list[str]) -> str:
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def select_label(
    ranked_labels: list[tuple[str, int]],
    *,
    requested: str | None,
    excluded: set[str],
    min_centroids: int = 1,
) -> str:
    if requested:
        for label, count in ranked_labels:
            if label == requested:
                if count < min_centroids:
                    raise ValueError(
                        f"Requested label '{requested}' has only {count} centroids; need at least {min_centroids}."
                    )
                return label
        raise ValueError(f"Requested label '{requested}' not found in filtered categories.")

    for label, count in ranked_labels:
        if label in excluded:
            continue
        if count >= min_centroids:
            return label
    raise ValueError(f"Could not find eligible label with at least {min_centroids} centroids.")


def write_hierarchy(output_dir: Path, cat_to_centroids: dict[str, list[int]], manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_labels = sorted(cat_to_centroids.keys())
    label_to_node_id = {label: idx for idx, label in enumerate(final_labels)}

    rows = []
    for label, centroids in cat_to_centroids.items():
        node_id = label_to_node_id[label]
        for centroid_id in sorted(int(c) for c in centroids):
            rows.append(
                {
                    "centroid_id": int(centroid_id),
                    "fcluster_label": int(node_id + 1),
                    "graph_node_id": int(node_id),
                }
            )
    assign_df = pd.DataFrame(rows).sort_values("centroid_id").reset_index(drop=True)
    assign_df.to_csv(output_dir / f"clusters_level_1_{len(final_labels)}_clusters.csv", index=False)

    labels_df = pd.DataFrame(
        [
            {
                "node_id": int(label_to_node_id[label]),
                "label": label,
                "description": f"Cluster for {label}",
            }
            for label in final_labels
        ]
    )
    labels_df.to_csv(output_dir / "inner_node_labels.csv", index=False)

    pd.DataFrame(
        {
            "threshold": [1.0],
            "n_clusters": [len(final_labels)],
            "silhouette_score": [0.0],
        }
    ).to_csv(output_dir / "optimal_thresholds.csv", index=False)

    manifest = dict(manifest)
    manifest["n_clusters"] = int(len(final_labels))
    manifest["cluster_sizes"] = {
        label: int(len(cat_to_centroids[label]))
        for label in final_labels
    }
    (output_dir / "corruption_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_variants(
    *,
    base_cat_to_centroids: dict[str, list[int]],
    ranked_labels: list[tuple[str, int]],
    rng: np.random.Generator,
    split_source_label: str | None,
    merge_keep_label: str | None,
    merge_drop_label: str | None,
    remove_source_label: str | None,
    add_source_label: str | None,
    add_target_labels: list[str] | None,
) -> dict[str, tuple[dict[str, list[int]], dict]]:
    variants: dict[str, tuple[dict[str, list[int]], dict]] = {}

    clean_map = {label: list(centroids) for label, centroids in base_cat_to_centroids.items()}
    variants["clean"] = (
        clean_map,
        {
            "mode": "clean",
            "expected_operation": "none",
            "notes": "Ground-truth clean hierarchy with no corruption.",
        },
    )

    split_source = select_label(
        ranked_labels,
        requested=split_source_label,
        excluded=set(),
        min_centroids=4,
    )
    merge_only_map = {label: list(centroids) for label, centroids in base_cat_to_centroids.items()}
    split_centroids = list(rng.permutation(merge_only_map.pop(split_source)))
    split_half = len(split_centroids) // 2
    merge_only_map[f"{split_source} core"] = split_centroids[:split_half]
    merge_only_map[f"{split_source} variant"] = split_centroids[split_half:]
    variants["merge_only"] = (
        merge_only_map,
        {
            "mode": "merge_only",
            "expected_operation": "merge",
            "split_source": split_source,
            "children": [f"{split_source} core", f"{split_source} variant"],
        },
    )

    merge_keep = select_label(
        ranked_labels,
        requested=merge_keep_label,
        excluded=set(),
        min_centroids=1,
    )
    merge_drop = select_label(
        ranked_labels,
        requested=merge_drop_label,
        excluded={merge_keep},
        min_centroids=1,
    )
    split_only_map = {label: list(centroids) for label, centroids in base_cat_to_centroids.items()}
    merged_label = f"{merge_keep} + {merge_drop}"
    split_only_map[merged_label] = split_only_map.pop(merge_keep) + split_only_map.pop(merge_drop)
    variants["split_only"] = (
        split_only_map,
        {
            "mode": "split_only",
            "expected_operation": "split",
            "merge_keep": merge_keep,
            "merge_drop": merge_drop,
            "merged_label": merged_label,
        },
    )

    remove_source = select_label(
        ranked_labels,
        requested=remove_source_label,
        excluded=set(),
        min_centroids=4,
    )
    remove_only_map = {label: list(centroids) for label, centroids in base_cat_to_centroids.items()}
    remove_centroids = list(rng.permutation(remove_only_map[remove_source]))
    orphan_count = max(1, min(2, len(remove_centroids) // 4))
    if len(remove_centroids) - orphan_count < 2:
        orphan_count = 1
    orphan_label = f"{remove_source} orphan"
    remove_only_map[remove_source] = remove_centroids[orphan_count:]
    remove_only_map[orphan_label] = remove_centroids[:orphan_count]
    variants["remove_only"] = (
        remove_only_map,
        {
            "mode": "remove_only",
            "expected_operation": "remove",
            "remove_source": remove_source,
            "orphan_label": orphan_label,
            "orphan_centroids": int(orphan_count),
        },
    )

    add_source = select_label(
        ranked_labels,
        requested=add_source_label,
        excluded=set(),
        min_centroids=4,
    )
    add_only_map = {label: list(centroids) for label, centroids in base_cat_to_centroids.items()}
    add_source_centroids = list(rng.permutation(add_only_map.pop(add_source)))
    if add_target_labels:
        if len(add_target_labels) != 2:
            raise ValueError("--add_target_labels must contain exactly 2 comma-separated labels.")
        target_a = select_label(
            ranked_labels,
            requested=add_target_labels[0],
            excluded={add_source},
            min_centroids=1,
        )
        target_b = select_label(
            ranked_labels,
            requested=add_target_labels[1],
            excluded={add_source, target_a},
            min_centroids=1,
        )
    else:
        target_a = select_label(
            ranked_labels,
            requested=None,
            excluded={add_source},
            min_centroids=1,
        )
        target_b = select_label(
            ranked_labels,
            requested=None,
            excluded={add_source, target_a},
            min_centroids=1,
        )
    add_half = len(add_source_centroids) // 2
    add_only_map[target_a].extend(add_source_centroids[:add_half])
    add_only_map[target_b].extend(add_source_centroids[add_half:])
    variants["add_only"] = (
        add_only_map,
        {
            "mode": "add_only",
            "expected_operation": "add",
            "hidden_source": add_source,
            "redistributed_to": [target_a, target_b],
        },
    )

    return variants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, required=True, help="CSV with centroid_id and true_label.")
    parser.add_argument("--output_root", type=str, required=True, help="Experiment directory root to write hierarchy_results__* subdirs.")
    parser.add_argument("--top_n_categories", type=int, default=8, help="Keep top-N true labels by centroid count.")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split_source_label", type=str, default=None)
    parser.add_argument("--merge_keep_label", type=str, default=None)
    parser.add_argument("--merge_drop_label", type=str, default=None)
    parser.add_argument("--remove_source_label", type=str, default=None)
    parser.add_argument("--add_source_label", type=str, default=None)
    parser.add_argument("--add_target_labels", type=str, default=None, help="Comma-separated pair of target labels for add_only mode.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data_csv = Path(args.data_csv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    required_cols = {"centroid_id", "true_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {data_csv}: {sorted(missing)}")

    centroid_to_label = (
        df.groupby("centroid_id")["true_label"]
        .agg(lambda s: plurality_label(list(s.astype(str))))
        .to_dict()
    )
    cat_to_centroids: dict[str, list[int]] = {}
    for centroid_id, label in centroid_to_label.items():
        cat_to_centroids.setdefault(str(label), []).append(int(centroid_id))

    ranked_labels = sorted(
        ((label, len(centroids)) for label, centroids in cat_to_centroids.items()),
        key=lambda item: (-item[1], item[0]),
    )
    ranked_labels = ranked_labels[: max(1, int(args.top_n_categories))]
    keep_labels = {label for label, _ in ranked_labels}
    base_cat_to_centroids = {
        label: sorted(cat_to_centroids[label])
        for label in keep_labels
    }
    ranked_labels = [
        (label, count)
        for label, count in ranked_labels
        if label in base_cat_to_centroids
    ]

    add_target_labels = None
    if args.add_target_labels:
        add_target_labels = [part.strip() for part in args.add_target_labels.split(",") if part.strip()]

    variants = build_variants(
        base_cat_to_centroids=base_cat_to_centroids,
        ranked_labels=ranked_labels,
        rng=rng,
        split_source_label=args.split_source_label,
        merge_keep_label=args.merge_keep_label,
        merge_drop_label=args.merge_drop_label,
        remove_source_label=args.remove_source_label,
        add_source_label=args.add_source_label,
        add_target_labels=add_target_labels,
    )

    summary = {
        "data_csv": str(data_csv),
        "seed": int(args.seed),
        "top_n_categories": int(args.top_n_categories),
        "base_cluster_sizes": {label: int(len(base_cat_to_centroids[label])) for label, _ in ranked_labels},
        "variants": {},
    }

    for mode, (cat_map, manifest) in variants.items():
        out_dir = output_root / f"hierarchy_results__{mode}"
        write_hierarchy(out_dir, cat_map, manifest)
        summary["variants"][mode] = {
            "output_dir": str(out_dir),
            "expected_operation": manifest.get("expected_operation"),
        }
        logging.info("Wrote %s to %s", mode, out_dir)

    summary_path = output_root / "newsgroups_sanity_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logging.info("Wrote summary manifest to %s", summary_path)


if __name__ == "__main__":
    main()
