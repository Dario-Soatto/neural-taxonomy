import argparse
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment_dir", required=True, help="e.g., experiments/wiki_biographies_1000")
    ap.add_argument("--clusters_file", required=True, help="clusters_level_*.csv from step 6")
    ap.add_argument("--text_col", default="description")
    ap.add_argument("--embedding_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    data_path = f"{args.experiment_dir}/models/all_extracted_discourse_with_clusters.csv"
    df = pd.read_csv(data_path)
    label_df = pd.read_csv(args.clusters_file)

    if "cluster" in df.columns:
        df = df.rename(columns={"cluster": "centroid_id"})
    merged = df.merge(label_df, on="centroid_id")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    embs = model.encode(merged[args.text_col].tolist(), convert_to_numpy=True)

    centroids = []
    cluster_ids = []
    for cid in sorted(merged["graph_node_id"].unique()):
        idx = merged.index[merged["graph_node_id"] == cid]
        centroids.append(embs[idx].mean(axis=0))
        cluster_ids.append(cid)

    sim = cosine_similarity(centroids)
    np.fill_diagonal(sim, -1)
    max_sim = sim.max()
    print("max cosine similarity between clusters:", max_sim)


if __name__ == "__main__":
    main()
