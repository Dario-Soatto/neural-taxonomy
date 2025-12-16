import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import argparse
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def load_data(dataset: str, filepath: str) -> pd.DataFrame:
    """Load dataset with ground truth labels."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)
            if dataset == 'bills':
                data.append({
                    'id': doc['id'],
                    'text': doc.get('summary', doc.get('tokenized_text', '')),
                    'label': doc.get('topic', ''),
                })
            else:  # wiki
                data.append({
                    'id': doc['id'],
                    'text': doc.get('text', ''),
                    'label': doc.get('supercategory', doc.get('category', '')),
                })
    return pd.DataFrame(data)


def compute_metrics(predicted, ground_truth) -> dict:
    """Compute ARI, NMI, Purity."""
    contingency = contingency_matrix(ground_truth, predicted)
    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
    return {
        'ARI': adjusted_rand_score(ground_truth, predicted),
        'NMI': normalized_mutual_info_score(ground_truth, predicted, average_method='arithmetic'),
        'Purity': purity,
    }


def run_embedding_kmeans(texts, n_clusters, random_state=42):
    """Method 1: Embedding + K-means."""
    print("  Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("  Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print("  Running K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(embeddings)


def run_lda(texts, n_topics, random_state=42):
    """Method 2: Classical LDA."""
    print("  Vectorizing text...")
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    print("  Running LDA...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method='online',
        max_iter=20,
        n_jobs=-1
    )
    doc_topics = lda.fit_transform(doc_term_matrix)
    return doc_topics.argmax(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['bills', 'wiki', 'both'], default='both')
    parser.add_argument('--sample_size', type=int, default=3000)
    args = parser.parse_args()

    datasets = []
    if args.dataset in ['bills', 'both']:
        datasets.append(('bills', 'src/train.metadata (2).jsonl'))
    if args.dataset in ['wiki', 'both']:
        datasets.append(('wiki', 'src/train.metadata (1).jsonl'))

    all_results = {}

    for dataset_name, filepath in datasets:
        print(f" DATASET: {dataset_name.upper()}")

        # Load data
        df = load_data(dataset_name, filepath)
        df = df[df['label'].notna() & (df['label'] != '') & df['text'].notna()]
        print(f"Loaded {len(df)} documents")

        # Sample
        if len(df) > args.sample_size:
            df = df.sample(n=args.sample_size, random_state=42)
            print(f"Sampled to {len(df)} documents")

        texts = df['text'].tolist()
        labels = df['label'].values
        n_classes = len(np.unique(labels))
        print(f"Ground truth classes: {n_classes}")

        results = {}

        # Method 1: Embedding + K-means
        print(f"\n[Method 1] Embedding + K-means")
        clusters_emb = run_embedding_kmeans(texts, n_classes)
        metrics_emb = compute_metrics(clusters_emb, labels)
        results['embedding_kmeans'] = metrics_emb
        print(f"  ARI: {metrics_emb['ARI']:.4f}  NMI: {metrics_emb['NMI']:.4f}  Purity: {metrics_emb['Purity']:.4f}")

        # Method 2: LDA
        print(f"\n[Method 2] LDA (Classical Topic Model)")
        clusters_lda = run_lda(texts, n_classes)
        metrics_lda = compute_metrics(clusters_lda, labels)
        results['lda'] = metrics_lda
        print(f"  ARI: {metrics_lda['ARI']:.4f}  NMI: {metrics_lda['NMI']:.4f}  Purity: {metrics_lda['Purity']:.4f}")

        all_results[dataset_name] = results

    # Summary table
    print(" SUMMARY: Baseline Comparison")
    print(f"{'Dataset':<10} {'Method':<20} {'ARI':>8} {'NMI':>8} {'Purity':>8}")
    for dataset_name, results in all_results.items():
        for method, metrics in results.items():
            method_name = "Embed+KMeans" if method == 'embedding_kmeans' else "LDA"
            print(f"{dataset_name:<10} {method_name:<20} {metrics['ARI']:>8.4f} {metrics['NMI']:>8.4f} {metrics['Purity']:>8.4f}")
    return all_results


if __name__ == "__main__":
    main()
