"""
Creates hierarchy_results/ files with a RANDOM cluster schema.
No text data is read — just generates the assignment structure.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_centroids", type=int, required=True,
                        help="Number of K-means centroids (IDs will be 0..n_centroids-1).")
    parser.add_argument("--n_clusters", type=int, default=10,
                        help="Number of random clusters to create.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the three hierarchy files into.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    centroid_ids = list(range(args.n_centroids))
    graph_node_ids = rng.integers(0, args.n_clusters, size=args.n_centroids).tolist()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cluster assignment file
    pd.DataFrame({
        "centroid_id":   centroid_ids,
        "fcluster_label": [g + 1 for g in graph_node_ids],
        "graph_node_id": graph_node_ids,
    }).to_csv(out_dir / f"clusters_level_1_{args.n_clusters}_clusters.csv", index=False)

    # 2. Cluster label file
    pd.DataFrame({
        "node_id":     list(range(args.n_clusters)),
        "label":       [f"topic_{i}" for i in range(args.n_clusters)],
        "description": [f"Random initial cluster {i}" for i in range(args.n_clusters)],
    }).to_csv(out_dir / "inner_node_labels.csv", index=False)

    # 3. Thresholds file (tells EM: 1 level, K clusters)
    pd.DataFrame({
        "threshold":        [1.0],
        "n_clusters":       [args.n_clusters],
        "silhouette_score": [0.0],
    }).to_csv(out_dir / "optimal_thresholds.csv", index=False)

    print(f"Written random clustering files to: {out_dir}")
    from collections import Counter
    dist = Counter(graph_node_ids)
    for k in sorted(dist):
        print(f"  topic_{k}: {dist[k]} centroids")


if __name__ == "__main__":
    main()