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
import os
from tqdm import tqdm
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_bills_data(filepath: str) -> pd.DataFrame:
    """Load Congressional Bills dataset with ground truth labels."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)
            data.append({
                'id': doc['id'],
                'text': doc.get('summary', doc.get('tokenized_text', '')),
                'label_high': doc.get('topic', ''),          # 21 high-level labels
                'label_low': doc.get('subtopic', ''),        # ~114 low-level labels
            })
    return pd.DataFrame(data)


def load_wiki_data(filepath: str) -> pd.DataFrame:
    """Load Wikipedia Biographies dataset with ground truth labels."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            doc = json.loads(line)
            data.append({
                'id': doc['id'],
                'text': doc.get('text', doc.get('tokenized_text', '')),
                'label_high': doc.get('supercategory', doc.get('category', '')),  # high-level
                'label_low': doc.get('subcategory', ''),                           # low-level
            })
    return pd.DataFrame(data)

def compute_ari(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Adjusted Rand Index: measures cluster agreement, adjusted for chance."""
    return adjusted_rand_score(ground_truth, predicted)


def compute_nmi(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Normalized Mutual Information: mutual dependence normalized to [0,1]."""
    return normalized_mutual_info_score(ground_truth, predicted, average_method='arithmetic')


def compute_purity(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Purity: fraction correctly assigned to majority class per cluster."""
    contingency = contingency_matrix(ground_truth, predicted)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def evaluate_clustering(predicted_clusters, ground_truth_labels) -> dict:
    """Compute all alignment metrics."""
    # Convert to numpy arrays and handle missing values
    mask = pd.notna(predicted_clusters) & pd.notna(ground_truth_labels)
    pred = np.array(predicted_clusters)[mask]
    true = np.array(ground_truth_labels)[mask]

    return {
        'ARI': compute_ari(pred, true),
        'NMI': compute_nmi(pred, true),
        'Purity': compute_purity(pred, true),
        'n_samples': len(pred),
        'n_clusters': len(np.unique(pred)),
        'n_true_classes': len(np.unique(true)),
    }

def run_baseline_clustering(
    texts: list,
    n_clusters: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    sample_size: int = None,
    random_state: int = 42
) -> tuple:
    """
    Run simple embedding + K-means clustering as a baseline.

    This is a simplified version of the neural-taxonomy pipeline for
    quick evaluation. The full pipeline adds:
    1. LLM-based initial labeling
    2. Fine-tuned embeddings from similarity pairs
    3. Hierarchical clustering

    Returns:
        clusters: cluster assignments for sampled texts
        sample_indices: indices of sampled texts in original list
    """
    print(f"\n{'='*60}")
    print(f"Running Baseline Clustering")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Clusters: {n_clusters}")

    n_total = len(texts)

    # Sample if dataset is large
    if sample_size and n_total > sample_size:
        print(f"Sampling {sample_size} from {n_total} documents...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_total, sample_size, replace=False)
        texts = [texts[i] for i in sample_indices]
    else:
        sample_indices = np.arange(n_total)

    # Load embedding model
    print(f"Loading embedding model...")
    model = SentenceTransformer(model_name)

    # Compute embeddings
    print(f"Computing embeddings for {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Run K-means
    print(f"Running K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    return clusters, sample_indices


def run_lda_baseline(
    texts: list,
    n_topics: int,
    sample_size: int = None,
    random_state: int = 42,
    max_features: int = 5000
) -> tuple:
    """
    Run classical LDA (Latent Dirichlet Allocation) as baseline.

    This is the classical topic model baseline from "Are Neural Topic Models Broken?"
    Similar to MALLET but using sklearn's implementation.
    """
    print(f"\n{'='*60}")
    print(f"Running LDA Baseline")
    print(f"{'='*60}")
    print(f"Topics: {n_topics}")

    n_total = len(texts)

    # Sample if dataset is large
    if sample_size and n_total > sample_size:
        print(f"Sampling {sample_size} from {n_total} documents...")
        np.random.seed(random_state)
        sample_indices = np.random.choice(n_total, sample_size, replace=False)
        texts = [texts[i] for i in sample_indices]
    else:
        sample_indices = np.arange(n_total)

    # Vectorize text
    print(f"Vectorizing {len(texts)} documents...")
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)

    # Run LDA
    print(f"Running LDA...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method='online',
        max_iter=20,
        n_jobs=-1
    )
    doc_topics = lda.fit_transform(doc_term_matrix)

    # Assign each document to its dominant topic
    clusters = doc_topics.argmax(axis=1)

    return clusters, sample_indices

def run_evaluation(
    dataset: str,
    data_path: str,
    n_clusters: int = None,
    sample_size: int = 5000,
    n_runs: int = 3,
    label_level: str = 'high'
):
    """
    Run full evaluation on a dataset.

    Args:
        dataset: 'bills' or 'wiki'
        data_path: path to JSONL file
        n_clusters: number of clusters (if None, uses ground truth count)
        sample_size: max samples to use (for speed)
        n_runs: number of runs for stability evaluation
        label_level: 'high' or 'low' for ground truth comparison
    """
    print(f"\n{'='*60}")
    print(f"EVALUATION BENCHMARK: {dataset.upper()}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading data from: {data_path}")
    if dataset == 'bills':
        df = load_bills_data(data_path)
    elif dataset == 'wiki':
        df = load_wiki_data(data_path)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    print(f"Loaded {len(df)} documents")

    # Select ground truth labels
    label_col = 'label_high' if label_level == 'high' else 'label_low'

    # Filter documents with valid labels and text
    df = df[df[label_col].notna() & (df[label_col] != '') & df['text'].notna()].copy()
    print(f"Documents with valid labels: {len(df)}")

    # Get ground truth stats
    true_labels = df[label_col].values
    n_true_classes = len(np.unique(true_labels))
    print(f"Ground truth classes ({label_level}-level): {n_true_classes}")

    # Set n_clusters to match ground truth if not specified
    if n_clusters is None:
        n_clusters = n_true_classes
        print(f"Using n_clusters = {n_clusters} (matching ground truth)")

    # Show label distribution
    label_counts = Counter(true_labels)
    print(f"\nTop 10 label distribution:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

    # Run multiple times for stability
    all_results = []
    all_clusters = []

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")

        # Run clustering
        clusters, sample_indices = run_baseline_clustering(
            df['text'].tolist(),
            n_clusters=n_clusters,
            sample_size=sample_size,
            random_state=42 + run_idx  # Different seed each run
        )

        # Evaluate on sampled subset
        sampled_clusters = clusters  # clusters are already for sampled texts
        sampled_labels = true_labels[sample_indices]

        results = evaluate_clustering(sampled_clusters, sampled_labels)
        all_results.append(results)
        all_clusters.append(sampled_clusters)

        print(f"\nRun {run_idx + 1} Results:")
        print(f"  ARI:    {results['ARI']:.4f}")
        print(f"  NMI:    {results['NMI']:.4f}")
        print(f"  Purity: {results['Purity']:.4f}")

    # Compute stability (pairwise ARI between runs)
    stability_scores = []
    for i in range(len(all_clusters)):
        for j in range(i + 1, len(all_clusters)):
            ari = adjusted_rand_score(all_clusters[i], all_clusters[j])
            stability_scores.append(ari)

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS: {dataset.upper()} ({label_level}-level labels)")
    print(f"{'='*60}")

    avg_ari = np.mean([r['ARI'] for r in all_results])
    avg_nmi = np.mean([r['NMI'] for r in all_results])
    avg_purity = np.mean([r['Purity'] for r in all_results])

    std_ari = np.std([r['ARI'] for r in all_results])
    std_nmi = np.std([r['NMI'] for r in all_results])
    std_purity = np.std([r['Purity'] for r in all_results])

    print(f"\nAlignment Metrics (mean ± std over {n_runs} runs):")
    print(f"  ARI:    {avg_ari:.4f} ± {std_ari:.4f}")
    print(f"  NMI:    {avg_nmi:.4f} ± {std_nmi:.4f}")
    print(f"  Purity: {avg_purity:.4f} ± {std_purity:.4f}")

    if stability_scores:
        print(f"\nStability (pairwise ARI between runs):")
        print(f"  Mean:   {np.mean(stability_scores):.4f}")
        print(f"  Std:    {np.std(stability_scores):.4f}")

    print(f"\nDataset Statistics:")
    print(f"  Total documents:     {len(df)}")
    print(f"  Sampled documents:   {all_results[0]['n_samples']}")
    print(f"  Predicted clusters:  {all_results[0]['n_clusters']}")
    print(f"  Ground truth classes: {all_results[0]['n_true_classes']}")

    # Return results dict
    return {
        'dataset': dataset,
        'label_level': label_level,
        'n_runs': n_runs,
        'alignment': {
            'ARI': {'mean': avg_ari, 'std': std_ari},
            'NMI': {'mean': avg_nmi, 'std': std_nmi},
            'Purity': {'mean': avg_purity, 'std': std_purity},
        },
        'stability': {
            'mean': np.mean(stability_scores) if stability_scores else None,
            'std': np.std(stability_scores) if stability_scores else None,
        },
        'per_run': all_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmark")
    parser.add_argument('--dataset', type=str, choices=['bills', 'wiki', 'both'],
                        default='both', help="Dataset to evaluate")
    parser.add_argument('--bills_path', type=str,
                        default='src/train.metadata (2).jsonl',
                        help="Path to Bills dataset")
    parser.add_argument('--wiki_path', type=str,
                        default='src/train.metadata (1).jsonl',
                        help="Path to Wiki dataset")
    parser.add_argument('--n_clusters', type=int, default=None,
                        help="Number of clusters (default: match ground truth)")
    parser.add_argument('--sample_size', type=int, default=5000,
                        help="Max samples to use per run")
    parser.add_argument('--n_runs', type=int, default=3,
                        help="Number of runs for stability")
    parser.add_argument('--label_level', type=str, choices=['high', 'low'],
                        default='high', help="Ground truth label level")
    parser.add_argument('--output_file', type=str, default=None,
                        help="Save results to JSON file")

    args = parser.parse_args()

    all_results = {}

    if args.dataset in ['bills', 'both']:
        results = run_evaluation(
            dataset='bills',
            data_path=args.bills_path,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size,
            n_runs=args.n_runs,
            label_level=args.label_level,
        )
        all_results['bills'] = results

    if args.dataset in ['wiki', 'both']:
        results = run_evaluation(
            dataset='wiki',
            data_path=args.wiki_path,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size,
            n_runs=args.n_runs,
            label_level=args.label_level,
        )
        all_results['wiki'] = results

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to: {args.output_file}")

    return all_results


if __name__ == "__main__":
    main()