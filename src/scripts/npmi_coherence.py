"""
Topic NPMI coherence (c_npmi) from cluster assignments + raw texts.

Uses Gensim's CoherenceModel with coherence='c_npmi'. Topic words are the
top unigrams by frequency within each cluster (post-hoc), not generative
beta from LDA—so baseline NPMI is comparable in *kind* across methods but
not identical to NPMI from inferred topic-word distributions.

Requires: pip install gensim
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, List, Optional

import pandas as pd

try:
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from gensim.utils import simple_preprocess

    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False


def _tokenize_corpus(texts: Iterable[str]) -> List[List[str]]:
    return [simple_preprocess(str(t), deacc=True, min_len=2) for t in texts]


def _topics_from_clusters(
    df: pd.DataFrame,
    text_col: str,
    cluster_col: str,
    topn: int = 10,
) -> List[List[str]]:
    """One topic = top `topn` word types by frequency in all docs of that cluster."""
    topics: List[List[str]] = []
    for cid in sorted(df[cluster_col].dropna().unique()):
        sub = df.loc[df[cluster_col] == cid, text_col].astype(str)
        blob = " ".join(sub.tolist())
        toks = simple_preprocess(blob, deacc=True, min_len=3)
        counts = Counter(toks)
        words = [w for w, _ in counts.most_common(topn)]
        topics.append(words if words else ["none"])
    return topics


def _filter_topics_to_dict(
    topics: List[List[str]], dictionary: Dictionary
) -> Optional[List[List[str]]]:
    tid = dictionary.token2id
    out: List[List[str]] = []
    for words in topics:
        kept = [w for w in words if w in tid]
        if kept:
            out.append(kept)
    return out if out else None


def npmi_c_npmi(
    texts_tokenized: List[List[str]],
    topics: List[List[str]],
    topn_words: int = 10,
) -> Optional[float]:
    if not _HAS_GENSIM:
        return None
    if not texts_tokenized or not topics:
        return None
    dictionary = Dictionary(texts_tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.95, keep_n=8000)
    if len(dictionary) < 5:
        return None
    topics_f = _filter_topics_to_dict(topics, dictionary)
    if not topics_f or len(topics_f) < 2:
        return None
    try:
        cm = CoherenceModel(
            topics=topics_f,
            texts=texts_tokenized,
            dictionary=dictionary,
            coherence="c_npmi",
            topn=min(topn_words, 10),
        )
        return float(cm.get_coherence())
    except Exception:
        return None


def npmi_from_doc_clusters(
    doc_assignments_csv: str,
    text_col: str = "text",
    cluster_col: str = "cluster_id",
    topn: int = 10,
) -> Optional[float]:
    df = pd.read_csv(doc_assignments_csv)
    if text_col not in df.columns or cluster_col not in df.columns:
        return None
    texts_tok = _tokenize_corpus(df[text_col].tolist())
    topics = _topics_from_clusters(df, text_col, cluster_col, topn=topn)
    return npmi_c_npmi(texts_tok, topics, topn_words=topn)


def npmi_from_text_cluster_csv(
    csv_path: str,
    text_col: str = "text",
    cluster_col: str = "cluster",
    topn: int = 10,
) -> Optional[float]:
    """Same as ``npmi_from_doc_clusters`` but default cluster column ``cluster`` (step 4 merge)."""
    return npmi_from_doc_clusters(csv_path, text_col, cluster_col, topn)


def npmi_from_centroid_mapping(
    merged_csv: str,
    level_clusters_csv: str,
    text_col: str = "text",
    centroid_col: str = "cluster",
    topn: int = 10,
) -> Optional[float]:
    """
    merged_csv: rows with document text and step-4 centroid id (`cluster`).
    level_clusters_csv: centroid_id -> fcluster_label (macro cluster).
    """
    if not _HAS_GENSIM:
        return None
    left = pd.read_csv(merged_csv)
    right = pd.read_csv(level_clusters_csv)
    if centroid_col not in left.columns or text_col not in left.columns:
        return None
    if "centroid_id" not in right.columns or "fcluster_label" not in right.columns:
        return None
    m = left.merge(
        right[["centroid_id", "fcluster_label"]],
        left_on=centroid_col,
        right_on="centroid_id",
        how="inner",
    )
    if m.empty:
        return None
    texts_tok = _tokenize_corpus(m[text_col].tolist())
    topics = _topics_from_clusters(m, text_col, "fcluster_label", topn=topn)
    return npmi_c_npmi(texts_tok, topics, topn_words=topn)
