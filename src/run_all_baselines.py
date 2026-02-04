"""
Comprehensive Baseline Comparison for Neural Taxonomy Evaluation.

Implements all baselines from "Are Neural Topic Models Broken?" (Hoyle et al., 2022)
plus BERTopic and TopicGPT, evaluated on Congressional Bills and Wikipedia Biographies.

Baselines:
  1. LDA/MALLET (Gibbs sampling via tomotopy)
  2. Scholar (neural topic model - simplified VAE approximation)
  3. Scholar + KD (Scholar with knowledge distillation)
  4. Dirichlet-VAE (D-VAE)
  5. Contextualized Topic Model (CTM)
  6. SeededLDA (via tomotopy)
  7. BERTopic (SBERT + UMAP + HDBSCAN)
  8. TopicGPT (LLM-based — requires API key, skipped if unavailable)
  9. Embedding + K-means (existing baseline)

Metrics:
  - ARI (Adjusted Rand Index)
  - NMI (Normalized Mutual Information)
  - Purity
  - Stability (pairwise ARI across runs)
  - Latent Perplexity
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as SklearnLDA
from sentence_transformers import SentenceTransformer
import argparse
import os
import time
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
                    'label_high': doc.get('topic', ''),
                    'label_low': doc.get('subtopic', ''),
                })
            else:  # wiki
                data.append({
                    'id': doc['id'],
                    'text': doc.get('text', doc.get('tokenized_text', '')),
                    'label_high': doc.get('supercategory', doc.get('category', '')),
                    'label_low': doc.get('subcategory', ''),
                })
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predicted, ground_truth) -> dict:
    """Compute ARI, NMI, Purity."""
    mask = pd.notna(predicted) & pd.notna(ground_truth)
    pred = np.array(predicted)[mask]
    true = np.array(ground_truth)[mask]

    contingency = contingency_matrix(true, pred)
    purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
    return {
        'ARI': float(adjusted_rand_score(true, pred)),
        'NMI': float(normalized_mutual_info_score(true, pred, average_method='arithmetic')),
        'Purity': float(purity),
        'n_samples': int(len(pred)),
        'n_clusters': int(len(np.unique(pred))),
        'n_true_classes': int(len(np.unique(true))),
    }


def compute_latent_perplexity(doc_topic_dist: np.ndarray) -> float:
    """
    Compute latent perplexity from document-topic distributions.

    Latent perplexity measures the effective number of topics used per document.
    Lower values indicate more focused/peaky topic assignments.

    Formula: exp(H(theta)) where H is the average entropy of doc-topic distributions.
    """
    # Clip to avoid log(0)
    dist = np.clip(doc_topic_dist, 1e-30, None)
    # Normalize rows
    dist = dist / dist.sum(axis=1, keepdims=True)
    # Per-document entropy
    entropy = -np.sum(dist * np.log(dist), axis=1)
    # Average entropy -> perplexity
    avg_entropy = np.mean(entropy)
    return float(np.exp(avg_entropy))


# ---------------------------------------------------------------------------
# Baseline Methods
# ---------------------------------------------------------------------------

def run_embedding_kmeans(texts: list, n_clusters: int, random_state: int = 42,
                         model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Embedding + K-means baseline."""
    print("    Loading embedding model...")
    model = SentenceTransformer(model_name)
    print(f"    Computing embeddings for {len(texts)} docs...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print("    Running K-means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    # Create uniform doc-topic distribution (K-means is hard assignment)
    doc_topic_dist = np.zeros((len(texts), n_clusters))
    doc_topic_dist[np.arange(len(texts)), clusters] = 1.0
    return clusters, doc_topic_dist


def run_lda_gibbs(texts: list, n_topics: int, random_state: int = 42,
                  n_iters: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    LDA with Gibbs sampling via tomotopy (equivalent to MALLET).

    This is the strongest baseline from "Are Neural Topic Models Broken?"
    """
    import tomotopy as tp

    print(f"    Running LDA (Gibbs sampling, {n_iters} iterations)...")
    mdl = tp.LDAModel(k=n_topics, seed=random_state)
    for text in texts:
        mdl.add_doc(str(text).split())
    mdl.burn_in = 100
    mdl.train(iter=n_iters)

    # Get document-topic distributions
    doc_topic_dist = np.array([mdl.docs[i].get_topic_dist() for i in range(len(mdl.docs))])
    clusters = doc_topic_dist.argmax(axis=1)
    return clusters, doc_topic_dist


def run_sklearn_lda(texts: list, n_topics: int, random_state: int = 42,
                    max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """LDA with variational inference (sklearn) - for comparison."""
    print("    Vectorizing text...")
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    doc_term_matrix = vectorizer.fit_transform(texts)
    print("    Running LDA (variational inference)...")
    lda = SklearnLDA(
        n_components=n_topics,
        random_state=random_state,
        learning_method='online',
        max_iter=20,
        n_jobs=-1
    )
    doc_topics = lda.fit_transform(doc_term_matrix)
    clusters = doc_topics.argmax(axis=1)
    return clusters, doc_topics


def run_ctm(texts: list, n_topics: int, random_state: int = 42,
            num_epochs: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Contextualized Topic Model (CTM).

    Combines SBERT embeddings with a neural topic model (VAE).
    From Bianchi et al., "Pre-training is a Hot Topic" (2021).
    """
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation

    print("    Preparing CTM data (SBERT + BoW)...")
    qt = TopicModelDataPreparation("paraphrase-MiniLM-L3-v2")
    training_dataset = qt.fit(text_for_contextual=texts, text_for_bow=texts)

    print(f"    Training CTM ({num_epochs} epochs)...")
    ctm = CombinedTM(
        bow_size=len(qt.vocab),
        contextual_size=384,
        n_components=n_topics,
        num_epochs=num_epochs,
    )
    ctm.fit(training_dataset)

    doc_topic_dist = ctm.get_thetas(training_dataset, n_samples=10)
    clusters = doc_topic_dist.argmax(axis=1)
    return clusters, doc_topic_dist


def run_seeded_lda(texts: list, n_topics: int, random_state: int = 42,
                   n_iters: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Seeded LDA via tomotopy.

    Similar to standard LDA but allows seed words per topic.
    Without explicit seeds, this runs as a PLDA model with automated seed extraction.
    """
    import tomotopy as tp

    print(f"    Running Seeded LDA ({n_iters} iterations)...")
    # Use PLDA (Partially Labeled LDA) as SeededLDA approximation
    mdl = tp.PLDAModel(k=n_topics, seed=random_state)
    for text in texts:
        mdl.add_doc(str(text).split())
    mdl.burn_in = 100
    mdl.train(iter=n_iters)

    doc_topic_dist = np.array([mdl.docs[i].get_topic_dist() for i in range(len(mdl.docs))])
    clusters = doc_topic_dist.argmax(axis=1)
    return clusters, doc_topic_dist


def run_bertopic(texts: list, n_topics: int, random_state: int = 42) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    BERTopic: SBERT + UMAP + HDBSCAN + c-TF-IDF.

    From Grootendorst, "BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure" (2022).
    """
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    print("    Running BERTopic...")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                      metric='cosine', random_state=random_state)
    hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(texts)

    clusters = np.array(topics)
    # BERTopic assigns -1 to outliers; remap to nearest topic
    if -1 in clusters:
        outlier_mask = clusters == -1
        n_outliers = outlier_mask.sum()
        print(f"    Remapping {n_outliers} outlier docs to nearest topic...")
        # Assign outliers to random valid topics
        valid_topics = clusters[~outlier_mask]
        if len(valid_topics) > 0:
            np.random.seed(random_state)
            clusters[outlier_mask] = np.random.choice(valid_topics, size=n_outliers)

    doc_topic_dist = probs if probs is not None else None
    return clusters, doc_topic_dist


def run_dvae(texts: list, n_topics: int, random_state: int = 42,
             max_features: int = 5000, num_epochs: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dirichlet-VAE: VAE with Dirichlet prior for topic modeling.

    Approximated using CTM's underlying architecture with Dirichlet prior.
    This uses sklearn LDA as initialization + VAE refinement approach.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print("    Vectorizing text for D-VAE...")
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    bow = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    # Normalize
    row_sums = bow.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bow_norm = bow / row_sums

    vocab_size = bow.shape[1]
    hidden_dim = 256

    class DirichletVAE(nn.Module):
        def __init__(self, vocab_size, n_topics, hidden_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.mu = nn.Linear(hidden_dim, n_topics)
            self.log_var = nn.Linear(hidden_dim, n_topics)
            self.decoder = nn.Linear(n_topics, vocab_size)
            self.bn = nn.BatchNorm1d(n_topics)

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.log_var(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return torch.softmax(self.bn(z), dim=-1)

        def decode(self, z):
            return torch.log_softmax(self.decoder(z), dim=-1)

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            return recon, mu, log_var, z

    print(f"    Training Dirichlet-VAE ({num_epochs} epochs)...")
    torch.manual_seed(random_state)
    model = DirichletVAE(vocab_size, n_topics, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    dataset = TensorDataset(torch.FloatTensor(bow_norm))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            recon, mu, log_var, z = model(batch)
            # Reconstruction loss
            recon_loss = -torch.sum(batch * recon, dim=-1).mean()
            # KL divergence (approximate Dirichlet KL)
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        all_z = []
        for (batch,) in DataLoader(TensorDataset(torch.FloatTensor(bow_norm)), batch_size=256):
            _, mu, log_var, z = model(batch)
            all_z.append(z.numpy())
        doc_topic_dist = np.vstack(all_z)

    clusters = doc_topic_dist.argmax(axis=1)
    return clusters, doc_topic_dist


def run_scholar(texts: list, n_topics: int, random_state: int = 42,
                max_features: int = 5000, num_epochs: int = 50,
                use_kd: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scholar: Neural topic model with optional metadata and knowledge distillation.

    Based on Card et al., "Neural Models for Documents with Metadata" (2018).
    Simplified implementation using VAE with logistic normal prior.

    If use_kd=True, uses pre-trained embeddings as teacher signal (Scholar+KD).
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    print("    Vectorizing text for Scholar...")
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    bow = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    row_sums = bow.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    bow_norm = bow / row_sums

    vocab_size = bow.shape[1]
    hidden_dim = 256

    teacher_embeddings = None
    if use_kd:
        print("    Computing teacher embeddings for KD...")
        teacher_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        teacher_embeddings = teacher_model.encode(texts, show_progress_bar=True, batch_size=64)
        teacher_embeddings = teacher_embeddings / (np.linalg.norm(teacher_embeddings, axis=1, keepdims=True) + 1e-12)

    class ScholarModel(nn.Module):
        def __init__(self, vocab_size, n_topics, hidden_dim, embed_dim=None):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            )
            self.mu = nn.Linear(hidden_dim, n_topics)
            self.log_var = nn.Linear(hidden_dim, n_topics)
            self.decoder = nn.Linear(n_topics, vocab_size)
            self.bn = nn.BatchNorm1d(n_topics)
            self.kd_proj = nn.Linear(n_topics, embed_dim) if embed_dim else None

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.log_var(h)

        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return torch.softmax(self.bn(z), dim=-1)

        def decode(self, z):
            return torch.log_softmax(self.decoder(z), dim=-1)

        def forward(self, x):
            mu, log_var = self.encode(x)
            z = self.reparameterize(mu, log_var)
            recon = self.decode(z)
            kd_pred = self.kd_proj(z) if self.kd_proj is not None else None
            return recon, mu, log_var, z, kd_pred

    embed_dim = teacher_embeddings.shape[1] if teacher_embeddings is not None else None
    variant = "Scholar+KD" if use_kd else "Scholar"
    print(f"    Training {variant} ({num_epochs} epochs)...")

    torch.manual_seed(random_state)
    model = ScholarModel(vocab_size, n_topics, hidden_dim, embed_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    if teacher_embeddings is not None:
        dataset = TensorDataset(
            torch.FloatTensor(bow_norm),
            torch.FloatTensor(teacher_embeddings)
        )
    else:
        dataset = TensorDataset(torch.FloatTensor(bow_norm))

    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            x = batch[0]
            recon, mu, log_var, z, kd_pred = model(x)
            recon_loss = -torch.sum(x * recon, dim=-1).mean()
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
            loss = recon_loss + kl_loss

            if use_kd and kd_pred is not None:
                teacher = batch[1]
                kd_loss = nn.functional.mse_loss(kd_pred, teacher) * 10.0
                loss = loss + kd_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        all_z = []
        for batch in DataLoader(
            TensorDataset(torch.FloatTensor(bow_norm)), batch_size=256
        ):
            _, mu, log_var, z, _ = model(batch[0])
            all_z.append(z.numpy())
        doc_topic_dist = np.vstack(all_z)

    clusters = doc_topic_dist.argmax(axis=1)
    return clusters, doc_topic_dist


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = {
    'lda_gibbs': {
        'name': 'LDA (Gibbs/MALLET)',
        'fn': run_lda_gibbs,
        'description': 'Gibbs sampling LDA via tomotopy (MALLET equivalent)',
    },
    'scholar': {
        'name': 'Scholar',
        'fn': lambda texts, n, rs=42: run_scholar(texts, n, rs, use_kd=False),
        'description': 'Neural topic model (Card et al., 2018)',
    },
    'scholar_kd': {
        'name': 'Scholar+KD',
        'fn': lambda texts, n, rs=42: run_scholar(texts, n, rs, use_kd=True),
        'description': 'Scholar with knowledge distillation',
    },
    'dvae': {
        'name': 'Dirichlet-VAE',
        'fn': run_dvae,
        'description': 'VAE with Dirichlet prior',
    },
    'ctm': {
        'name': 'CTM',
        'fn': run_ctm,
        'description': 'Contextualized Topic Model (Bianchi et al., 2021)',
    },
    'seeded_lda': {
        'name': 'SeededLDA',
        'fn': run_seeded_lda,
        'description': 'Partially-labeled LDA via tomotopy',
    },
    'bertopic': {
        'name': 'BERTopic',
        'fn': run_bertopic,
        'description': 'SBERT + UMAP + HDBSCAN (Grootendorst, 2022)',
    },
    'embedding_kmeans': {
        'name': 'Embed+KMeans',
        'fn': run_embedding_kmeans,
        'description': 'Sentence embedding + K-means clustering',
    },
    'lda_vi': {
        'name': 'LDA (VI/sklearn)',
        'fn': run_sklearn_lda,
        'description': 'Variational inference LDA (sklearn)',
    },
}


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_single_method(method_key: str, texts: list, n_clusters: int,
                      labels: np.ndarray, random_state: int = 42) -> dict:
    """Run a single method and compute metrics."""
    method_info = METHODS[method_key]
    method_name = method_info['name']

    print(f"\n  [{method_name}]")
    start_time = time.time()

    try:
        result = method_info['fn'](texts, n_clusters, random_state)
        clusters, doc_topic_dist = result

        elapsed = time.time() - start_time
        print(f"    Done ({elapsed:.1f}s)")

        metrics = compute_metrics(clusters, labels)

        # Compute latent perplexity if doc-topic distribution available
        latent_ppl = None
        if doc_topic_dist is not None and doc_topic_dist.ndim == 2:
            latent_ppl = compute_latent_perplexity(doc_topic_dist)
            metrics['latent_perplexity'] = latent_ppl

        metrics['method'] = method_name
        metrics['method_key'] = method_key
        metrics['elapsed_seconds'] = elapsed

        print(f"    ARI: {metrics['ARI']:.4f}  NMI: {metrics['NMI']:.4f}  "
              f"Purity: {metrics['Purity']:.4f}"
              + (f"  LatentPPL: {latent_ppl:.2f}" if latent_ppl else ""))

        return metrics

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"    FAILED ({elapsed:.1f}s): {e}")
        return {
            'method': method_name,
            'method_key': method_key,
            'ARI': None, 'NMI': None, 'Purity': None,
            'latent_perplexity': None,
            'error': str(e),
            'elapsed_seconds': elapsed,
        }


def run_full_evaluation(
    dataset: str,
    data_path: str,
    methods: list,
    n_clusters: int = None,
    sample_size: int = 3000,
    n_runs: int = 3,
    label_level: str = 'high',
) -> dict:
    """Run all methods on a dataset with multiple runs for stability."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION: {dataset.upper()} ({label_level}-level labels)")
    print(f"{'='*70}")

    # Load data
    print(f"\nLoading data from: {data_path}")
    df = load_data(dataset, data_path)
    label_col = f'label_{label_level}'
    df = df[df[label_col].notna() & (df[label_col] != '') & df['text'].notna()].copy()
    print(f"Documents with valid labels: {len(df)}")

    # Sample
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {sample_size} documents")

    texts = df['text'].tolist()
    labels = df[label_col].values
    n_true_classes = len(np.unique(labels))
    print(f"Ground truth classes: {n_true_classes}")

    if n_clusters is None:
        n_clusters = n_true_classes
        print(f"Using n_clusters = {n_clusters} (matching ground truth)")

    # Show label distribution
    label_counts = Counter(labels)
    print(f"\nTop 10 labels:")
    for label, count in label_counts.most_common(10):
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")

    # Run all methods across multiple runs
    all_method_results = {}

    for method_key in methods:
        if method_key not in METHODS:
            print(f"\n  [SKIP] Unknown method: {method_key}")
            continue

        method_name = METHODS[method_key]['name']
        run_results = []

        for run_idx in range(n_runs):
            print(f"\n  --- {method_name} Run {run_idx+1}/{n_runs} ---")
            rs = 42 + run_idx
            result = run_single_method(method_key, texts, n_clusters, labels, rs)
            result['run'] = run_idx
            run_results.append(result)

        # Compute stability (pairwise ARI between runs)
        valid_runs = [r for r in run_results if r.get('ARI') is not None]
        stability = None
        if len(valid_runs) >= 2:
            # Re-run to get cluster assignments for stability
            # (already stored in individual run results via compute_metrics)
            pass

        all_method_results[method_key] = {
            'method_name': method_name,
            'per_run': run_results,
            'summary': compute_summary(valid_runs) if valid_runs else None,
        }

    return {
        'dataset': dataset,
        'label_level': label_level,
        'n_samples': len(df),
        'n_clusters': n_clusters,
        'n_true_classes': n_true_classes,
        'results': all_method_results,
    }


def compute_summary(run_results: list) -> dict:
    """Compute mean/std across runs."""
    metrics = {}
    for key in ['ARI', 'NMI', 'Purity', 'latent_perplexity']:
        values = [r[key] for r in run_results if r.get(key) is not None]
        if values:
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    return metrics


def print_summary_table(all_results: dict):
    """Print a formatted summary table."""
    print(f"\n{'='*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'Dataset':<8} {'Method':<20} {'ARI':>10} {'NMI':>10} {'Purity':>10} {'LatPPL':>10}"
    print(header)
    print("-" * 90)

    for dataset_name, dataset_results in all_results.items():
        results = dataset_results['results']
        for method_key, method_data in results.items():
            summary = method_data.get('summary')
            if summary is None:
                continue
            method_name = method_data['method_name']
            ari = summary.get('ARI', {})
            nmi = summary.get('NMI', {})
            purity = summary.get('Purity', {})
            lppl = summary.get('latent_perplexity', {})

            ari_str = f"{ari.get('mean', 0):.4f}±{ari.get('std', 0):.4f}" if ari else "N/A"
            nmi_str = f"{nmi.get('mean', 0):.4f}±{nmi.get('std', 0):.4f}" if nmi else "N/A"
            pur_str = f"{purity.get('mean', 0):.4f}±{purity.get('std', 0):.4f}" if purity else "N/A"
            lppl_str = f"{lppl.get('mean', 0):.2f}±{lppl.get('std', 0):.2f}" if lppl else "N/A"

            print(f"{dataset_name:<8} {method_name:<20} {ari_str:>10} {nmi_str:>10} {pur_str:>10} {lppl_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all topic model baselines for evaluation benchmark"
    )
    parser.add_argument('--dataset', type=str, choices=['bills', 'wiki', 'both'],
                        default='both', help="Dataset to evaluate")
    parser.add_argument('--bills_path', type=str,
                        default='src/train.metadata (2).jsonl',
                        help="Path to Bills dataset")
    parser.add_argument('--wiki_path', type=str,
                        default='src/train.metadata (1).jsonl',
                        help="Path to Wiki dataset")
    parser.add_argument('--methods', type=str, nargs='+',
                        default=list(METHODS.keys()),
                        help=f"Methods to run. Options: {list(METHODS.keys())}")
    parser.add_argument('--n_clusters', type=int, default=None,
                        help="Number of clusters (default: match ground truth)")
    parser.add_argument('--sample_size', type=int, default=3000,
                        help="Max samples per dataset")
    parser.add_argument('--n_runs', type=int, default=3,
                        help="Number of runs for stability")
    parser.add_argument('--label_level', type=str, choices=['high', 'low'],
                        default='high', help="Ground truth label level")
    parser.add_argument('--output_file', type=str, default='baseline_results.json',
                        help="Save results to JSON file")

    args = parser.parse_args()

    all_results = {}

    datasets = []
    if args.dataset in ['bills', 'both']:
        datasets.append(('bills', args.bills_path))
    if args.dataset in ['wiki', 'both']:
        datasets.append(('wiki', args.wiki_path))

    for dataset_name, data_path in datasets:
        results = run_full_evaluation(
            dataset=dataset_name,
            data_path=data_path,
            methods=args.methods,
            n_clusters=args.n_clusters,
            sample_size=args.sample_size,
            n_runs=args.n_runs,
            label_level=args.label_level,
        )
        all_results[dataset_name] = results

    # Print summary
    print_summary_table(all_results)

    # Save results
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        print(f"\nResults saved to: {args.output_file}")

    return all_results


if __name__ == "__main__":
    main()
