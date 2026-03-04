"""
Prepares the HuffPost News Category dataset for the EM algorithm.

Dataset: heegyu/news-category-dataset (~210k headlines, 41 categories)
Each example has a short headline (~60 chars) + one-sentence description.
We concatenate them into a single text field.

Steps:
  1. Load the dataset from HuggingFace.
  2. Subsample to --n_docs documents (for speed).
  3. Embed with sentence-transformers/all-MiniLM-L6-v2.
  4. K-means to --n_centroids micro-clusters.
  5. Write a CSV with columns: centroid_id, description, true_label
     (true_label is kept for evaluation but is NOT used by the EM).

Usage (on cluster):
  python src/scripts/prepare_huffpost_data.py \
      --output_csv /path/to/huffpost_data.csv \
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── 1. Load HuffPost News Category dataset ───────────────────────────────
    logging.info("Loading HuffPost News Category dataset...")
    from datasets import load_dataset
    dataset = load_dataset("heegyu/news-category-dataset", split="train")
    logging.info("Loaded %d examples across categories.", len(dataset))

    # Combine headline + short_description into a single text
    texts = []
    categories = []
    for ex in dataset:
        headline = (ex.get("headline") or "").strip()
        desc = (ex.get("short_description") or "").strip()
        if headline and desc:
            text = f"{headline}. {desc}"
        elif headline:
            text = headline
        else:
            continue
        if len(text) >= 20:
            texts.append(text)
            categories.append(ex["category"])

    logging.info("%d usable examples with %d unique categories.",
                 len(texts), len(set(categories)))
    logging.info("Category distribution (top 20):\n%s",
                 pd.Series(categories).value_counts().head(20).to_string())

    # ── 2. Subsample ─────────────────────────────────────────────────────────
    n = min(args.n_docs, len(texts))
    idx = rng.choice(len(texts), size=n, replace=False)
    texts = [texts[i] for i in idx]
    categories = [categories[i] for i in idx]
    logging.info("Subsampled to %d documents.", n)
    logging.info("Category distribution after subsampling:\n%s",
                 pd.Series(categories).value_counts().to_string())

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
    logging.info("K-means done. Centroid id range: %d–%d.",
                 centroid_ids.min(), centroid_ids.max())

    # ── 5. Write CSV ─────────────────────────────────────────────────────────
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "centroid_id": centroid_ids,
        "description": texts,
        "true_label": categories,
    })
    df.to_csv(out_path, index=False)
    logging.info("Wrote %d rows to %s.", len(df), out_path)


if __name__ == "__main__":
    main()
