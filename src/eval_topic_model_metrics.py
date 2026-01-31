import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from collections import Counter
from typing import List, Dict, Tuple, Optional
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def compute_ari(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index between cluster assignments and ground truth.

    ARI measures the similarity between two clusterings, adjusted for chance.
    Range: [-1, 1], where 1 = perfect match, 0 = random, <0 = worse than random

    Args:
        cluster_labels: Predicted cluster assignments
        ground_truth: True labels

    Returns:
        ARI score
    """
    return adjusted_rand_score(ground_truth, cluster_labels)


def compute_nmi(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Normalized Mutual Information between clusters and ground truth.

    NMI measures the mutual dependence between clusterings, normalized to [0, 1].
    Range: [0, 1], where 1 = perfect match, 0 = independent

    Args:
        cluster_labels: Predicted cluster assignments
        ground_truth: True labels

    Returns:
        NMI score
    """
    return normalized_mutual_info_score(ground_truth, cluster_labels, average_method='arithmetic')


def compute_purity(cluster_labels: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute cluster purity - fraction of correctly assigned samples.

    For each cluster, assigns all members to the most frequent ground truth class.
    Range: [0, 1], where 1 = perfectly pure clusters

    Note: Purity can be trivially maximized by having one cluster per sample,
    so it should be used alongside other metrics.

    Args:
        cluster_labels: Predicted cluster assignments
        ground_truth: True labels

    Returns:
        Purity score
    """
    contingency = contingency_matrix(ground_truth, cluster_labels)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


def compute_all_alignment_metrics(
    cluster_labels: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, float]:
    """
    Compute all alignment metrics at once.

    Args:
        cluster_labels: Predicted cluster assignments
        ground_truth: True labels

    Returns:
        Dictionary with ARI, NMI, and Purity scores
    """
    return {
        "ARI": compute_ari(cluster_labels, ground_truth),
        "NMI": compute_nmi(cluster_labels, ground_truth),
        "Purity": compute_purity(cluster_labels, ground_truth),
    }

def rbo_score(
    list1: List,
    list2: List,
    p: float = 0.9
) -> float:
    """
    Compute Rank-Biased Overlap (RBO) between two ranked lists.

    RBO measures the similarity of two ranked lists, with higher weight
    given to items at the top of the list. Parameter p controls the
    top-heaviness of the metric.

    Based on: Webber et al., "A Similarity Measure for Indefinite Rankings" (2010)

    Args:
        list1: First ranked list
        list2: Second ranked list
        p: Persistence parameter (0 < p < 1). Higher p = more weight to deep ranks.
           p=0.9 means top 10 items get ~86% of the weight.

    Returns:
        RBO score in [0, 1], where 1 = identical rankings
    """
    if len(list1) == 0 and len(list2) == 0:
        return 1.0
    if len(list1) == 0 or len(list2) == 0:
        return 0.0

    # Convert to sets for overlap calculation
    set1 = set()
    set2 = set()

    # Compute RBO with extrapolation
    depth = max(len(list1), len(list2))
    rbo_sum = 0.0

    for d in range(1, depth + 1):
        if d <= len(list1):
            set1.add(list1[d-1])
        if d <= len(list2):
            set2.add(list2[d-1])

        # Agreement at depth d
        overlap = len(set1 & set2)
        agreement = overlap / d

        # Weight by p^(d-1)
        rbo_sum += (p ** (d - 1)) * agreement

    return (1 - p) * rbo_sum


def get_topic_word_ranking(
    cluster_labels: np.ndarray,
    texts: List[str],
    top_n: int = 20
) -> Dict[int, List[str]]:
    """
    Get top words for each cluster/topic based on frequency.

    Args:
        cluster_labels: Cluster assignments
        texts: Text documents
        top_n: Number of top words to return per cluster

    Returns:
        Dictionary mapping cluster ID to ranked list of top words
    """
    from collections import defaultdict

    cluster_words = defaultdict(list)
    for label, text in zip(cluster_labels, texts):
        words = str(text).lower().split()
        cluster_words[label].extend(words)

    topic_rankings = {}
    for cluster_id, words in cluster_words.items():
        word_counts = Counter(words)
        # Remove common stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                     'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just',
                     'that', 'this', 'these', 'those', 'it', 'its', 'they', 'their'}
        for sw in stopwords:
            word_counts.pop(sw, None)
        top_words = [word for word, _ in word_counts.most_common(top_n)]
        topic_rankings[cluster_id] = top_words

    return topic_rankings


def compute_stability_rbo(
    runs: List[Dict[int, List[str]]],
    p: float = 0.9
) -> Dict[str, float]:
    """
    Compute RBO-based stability across multiple runs.

    Matches topics across runs using Hungarian algorithm on RBO matrix,
    then computes average RBO for matched topics.

    Args:
        runs: List of topic word rankings from different runs
        p: RBO persistence parameter

    Returns:
        Dictionary with stability metrics
    """
    if len(runs) < 2:
        return {"stability_rbo": None, "message": "Need at least 2 runs for stability"}

    from scipy.optimize import linear_sum_assignment

    all_pairwise_rbo = []

    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            run1 = runs[i]
            run2 = runs[j]

            # Build RBO cost matrix (we use 1 - RBO for minimization)
            topics1 = list(run1.keys())
            topics2 = list(run2.keys())

            n1, n2 = len(topics1), len(topics2)
            cost_matrix = np.zeros((n1, n2))

            for ti, t1 in enumerate(topics1):
                for tj, t2 in enumerate(topics2):
                    rbo = rbo_score(run1[t1], run2[t2], p=p)
                    cost_matrix[ti, tj] = 1 - rbo  # Convert to cost

            # Hungarian algorithm for optimal matching
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Compute average RBO for matched topics
            matched_rbos = [1 - cost_matrix[r, c] for r, c in zip(row_ind, col_ind)]
            avg_rbo = np.mean(matched_rbos) if matched_rbos else 0.0
            all_pairwise_rbo.append(avg_rbo)

    return {
        "stability_rbo_mean": float(np.mean(all_pairwise_rbo)),
        "stability_rbo_std": float(np.std(all_pairwise_rbo)),
        "stability_rbo_min": float(np.min(all_pairwise_rbo)),
        "stability_rbo_max": float(np.max(all_pairwise_rbo)),
        "n_comparisons": len(all_pairwise_rbo),
    }


def compute_cluster_assignment_stability(
    runs: List[np.ndarray]
) -> Dict[str, float]:
    """
    Compute stability based on cluster assignment consistency across runs.

    Uses pairwise ARI between cluster assignments from different runs.

    Args:
        runs: List of cluster assignment arrays from different runs

    Returns:
        Dictionary with stability metrics
    """
    if len(runs) < 2:
        return {"assignment_stability_ari": None}

    pairwise_ari = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            ari = adjusted_rand_score(runs[i], runs[j])
            pairwise_ari.append(ari)

    return {
        "assignment_stability_ari_mean": float(np.mean(pairwise_ari)),
        "assignment_stability_ari_std": float(np.std(pairwise_ari)),
        "assignment_stability_ari_min": float(np.min(pairwise_ari)),
        "assignment_stability_ari_max": float(np.max(pairwise_ari)),
    }

def evaluate_clustering(
    df: pd.DataFrame,
    cluster_col: str,
    label_col: str,
    text_col: Optional[str] = None,
    compute_topic_coherence: bool = False
) -> Dict:
    """
    Run full evaluation on a clustering result.

    Args:
        df: DataFrame with cluster assignments and ground truth labels
        cluster_col: Column name for cluster assignments
        label_col: Column name for ground truth labels
        text_col: Column name for text (optional, for topic word extraction)
        compute_topic_coherence: Whether to compute topic coherence metrics

    Returns:
        Dictionary with all evaluation metrics
    """
    mask = df[cluster_col].notna() & df[label_col].notna()
    df_clean = df[mask].copy()

    cluster_labels = df_clean[cluster_col].values
    ground_truth = df_clean[label_col].values

    results = {
        "n_samples": len(df_clean),
        "n_clusters": len(np.unique(cluster_labels)),
        "n_true_classes": len(np.unique(ground_truth)),
    }

    alignment = compute_all_alignment_metrics(cluster_labels, ground_truth)
    results.update(alignment)

    if text_col is not None and text_col in df_clean.columns:
        topic_rankings = get_topic_word_ranking(
            cluster_labels,
            df_clean[text_col].tolist()
        )
        results["topic_rankings"] = {str(k): v for k, v in topic_rankings.items()}

    return results


def evaluate_stability_from_files(
    cluster_files: List[str],
    cluster_col: str,
    text_col: Optional[str] = None
) -> Dict:
    """
    Evaluate stability across multiple run files.

    Args:
        cluster_files: List of paths to cluster result files
        cluster_col: Column name for cluster assignments
        text_col: Column name for text (for RBO calculation)

    Returns:
        Dictionary with stability metrics
    """
    cluster_assignments = []
    topic_rankings = []

    for fpath in cluster_files:
        if fpath.endswith('.csv'):
            df = pd.read_csv(fpath)
        else:
            df = pd.read_json(fpath, lines=True)

        cluster_assignments.append(df[cluster_col].values)

        if text_col is not None and text_col in df.columns:
            rankings = get_topic_word_ranking(
                df[cluster_col].values,
                df[text_col].tolist()
            )
            topic_rankings.append(rankings)

    results = compute_cluster_assignment_stability(cluster_assignments)

    if topic_rankings:
        rbo_results = compute_stability_rbo(topic_rankings)
        results.update(rbo_results)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate topic model clustering with alignment and stability metrics"
    )
    parser.add_argument(
        "--cluster_file",
        type=str,
        help="Path to file with cluster assignments"
    )
    parser.add_argument(
        "--cluster_files",
        type=str,
        nargs="+",
        help="Multiple files for stability evaluation"
    )
    parser.add_argument(
        "--cluster_col",
        type=str,
        required=True,
        help="Column name for cluster assignments"
    )
    parser.add_argument(
        "--label_col",
        type=str,
        help="Column name for ground truth labels"
    )
    parser.add_argument(
        "--text_col",
        type=str,
        help="Column name for text (for topic word extraction)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Path to save evaluation results (JSON)"
    )
    parser.add_argument(
        "--compute_stability",
        action="store_true",
        help="Compute stability metrics across multiple runs"
    )

    args = parser.parse_args()

    results = {}

    if args.cluster_file:
        logging.info(f"Loading cluster file: {args.cluster_file}")
        if args.cluster_file.endswith('.csv'):
            df = pd.read_csv(args.cluster_file)
        else:
            df = pd.read_json(args.cluster_file, lines=True)

        if args.label_col:
            logging.info("Computing alignment metrics...")
            results["alignment"] = evaluate_clustering(
                df,
                cluster_col=args.cluster_col,
                label_col=args.label_col,
                text_col=args.text_col
            )
            logging.info(f"ARI: {results['alignment']['ARI']:.4f}")
            logging.info(f"NMI: {results['alignment']['NMI']:.4f}")
            logging.info(f"Purity: {results['alignment']['Purity']:.4f}")

    if args.compute_stability and args.cluster_files:
        logging.info(f"Computing stability across {len(args.cluster_files)} runs...")
        results["stability"] = evaluate_stability_from_files(
            args.cluster_files,
            cluster_col=args.cluster_col,
            text_col=args.text_col
        )
        logging.info(f"Stability (ARI): {results['stability'].get('assignment_stability_ari_mean', 'N/A')}")
        if 'stability_rbo_mean' in results['stability']:
            logging.info(f"Stability (RBO): {results['stability']['stability_rbo_mean']:.4f}")

    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logging.info(f"Results saved to: {args.output_file}")
    else:
        print(json.dumps(results, indent=2, default=str))

    return results


if __name__ == "__main__":
    main()
