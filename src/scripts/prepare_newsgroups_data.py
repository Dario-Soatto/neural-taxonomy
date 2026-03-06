"""
Prepares a 20 Newsgroups sentence-data CSV for the EM algorithm.

Steps:
  1. Load 20 Newsgroups from sklearn (no manual download needed).
  2. Subsample to --n_docs documents (for speed).
  3. Embed with sentence-transformers/all-MiniLM-L6-v2.
  4. K-means to --n_centroids clusters.
  5. Write a CSV with columns: centroid_id, description, true_label
     (true_label is kept for evaluation but is NOT used by the EM).

Usage (on cluster):
  python src/scripts/prepare_newsgroups_data.py \
      --output_csv /path/to/newsgroups_data.csv \
      --n_docs 1000 \
      --n_centroids 200 \
      --seed 42
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to write the output CSV.")
    parser.add_argument("--n_docs", type=int, default=1000,
                        help="Number of documents to sample (default: 1000).")
    parser.add_argument("--n_centroids", type=int, default=200,
                        help="Number of K-means centroids (default: 200).")
    parser.add_argument("--embedding_model", type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="SentenceTransformer model for embeddings.")
    parser.add_argument("--subset", type=str, default="all",
                        choices=["train", "test", "all"],
                        help="Which split of 20 Newsgroups to use.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── 1. Load 20 Newsgroups ────────────────────────────────────────────────
    logging.info("Loading 20 Newsgroups dataset (subset='%s')...", args.subset)
    from sklearn.datasets import fetch_20newsgroups
    dataset = fetch_20newsgroups(
        subset=args.subset,
        remove=("headers", "footers", "quotes"),  # strip metadata for cleaner text
        random_state=args.seed,
    )
    texts = dataset.data
    true_labels = [dataset.target_names[t] for t in dataset.target]
    logging.info("Loaded %d documents across %d categories.", len(texts), len(dataset.target_names))

    # ── 2. Subsample ─────────────────────────────────────────────────────────
    n = min(args.n_docs, len(texts))
    idx = rng.choice(len(texts), size=n, replace=False)
    texts = [texts[i] for i in idx]
    true_labels = [true_labels[i] for i in idx]
    logging.info("Subsampled to %d documents.", n)

    # Clean up: strip whitespace, drop empty docs
    cleaned = []
    kept_labels = []
    for txt, lbl in zip(texts, true_labels):
        txt = txt.strip()
        if len(txt) >= 20:
            cleaned.append(txt)
            kept_labels.append(lbl)
    texts, true_labels = cleaned, kept_labels
    logging.info("%d documents remain after cleaning.", len(texts))

    # ── 3. Embed ─────────────────────────────────────────────────────────────
    logging.info("Loading SentenceTransformer '%s'...", args.embedding_model)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.embedding_model)
    logging.info("Encoding %d documents...", len(texts))
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True)
    logging.info("Embeddings shape: %s", embeddings.shape)

    # ── 4. K-means ───────────────────────────────────────────────────────────
    n_centroids = min(args.n_centroids, len(texts))
    logging.info("Running K-means with %d centroids...", n_centroids)
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=n_centroids,
        random_state=args.seed,
        n_init=3,
        batch_size=1024,
    )
    centroid_ids = kmeans.fit_predict(embeddings)
    logging.info("K-means done. Centroid id range: %d–%d.", centroid_ids.min(), centroid_ids.max())

    # ── 5. Write CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "centroid_id": centroid_ids,
        "description": texts,
        "true_label": true_labels,
    })
    df.to_csv(out_path, index=False)
    logging.info("Wrote %d rows to %s.", len(df), out_path)
    logging.info("Centroid distribution (first 10):\n%s",
                 df["centroid_id"].value_counts().head(10).to_string())
    logging.info("True label distribution:\n%s",
                 df["true_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
