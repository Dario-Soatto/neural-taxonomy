from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class SchemaState:
    cluster_ids: List[int]
    label_map: Dict[int, str]
    desc_map: Dict[int, str] | None = None
    p_z_prior: np.ndarray | None = None

    @property
    def choices_list(self) -> List[str]:
        return [self.label_map[cid] for cid in self.cluster_ids]

    def rebuild_prior_from_df(self, df: pd.DataFrame, label_col: str = "agglomerative_label") -> None:
        counts = df[label_col].astype(str).value_counts()
        pz = np.array([counts.get(lbl, 0) for lbl in self.choices_list], dtype=float)
        if pz.sum() > 0:
            pz = pz / pz.sum()
        else:
            pz = np.ones(len(self.choices_list)) / max(len(self.choices_list), 1)
        self.p_z_prior = pz


def build_schema_state_from_df(df: pd.DataFrame) -> SchemaState:
    cluster_ids = sorted(df["cluster_id"].unique())
    label_map = (
        df.groupby("cluster_id")["agglomerative_label"]
        .first()
        .astype(str)
        .map(lambda x: build_label_string(x))
        .to_dict()
    )
    schema = SchemaState(cluster_ids=cluster_ids, label_map=label_map)
    schema.rebuild_prior_from_df(df)
    return schema


def sanitize_text(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    s = " ".join(s.split())
    return s


def base_label(name: str) -> str:
    s = sanitize_text(name)
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s


def dedupe_tokens(tokens: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def build_label_string(name: str, keywords: list[str] | None = None) -> str:
    name = base_label(name)
    if not name:
        name = "Misc"
    name = name[:40]
    if keywords:
        kws = [sanitize_text(k) for k in keywords if sanitize_text(k)]
        kws = dedupe_tokens(kws)[:8]
        if kws:
            kw_str = " ".join(kws)
            name = f"{name}: {kw_str}"
    return name[:160]


def propose_label_from_texts(
    texts: List[str],
    parent_label: str | None = None,
    max_keywords: int = 2,
) -> str:
    """
    Deterministic label heuristic using TF-IDF keywords.
    Returns a short label string.
    """
    if not texts:
        return parent_label or "Misc"

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500,
        )
        X = vectorizer.fit_transform(texts)
        if X.shape[1] == 0:
            return parent_label or "Misc"
        scores = np.asarray(X.mean(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:max_keywords]
        terms = [vectorizer.get_feature_names_out()[i] for i in top_idx if scores[i] > 0]
    except Exception:
        terms = []

    if not terms:
        return parent_label or "Misc"

    kw = " ".join(terms)
    if parent_label:
        return f"{parent_label}: {kw}"
    return f"Topic: {kw}"


def propose_label_and_keywords_from_texts(
    texts: List[str],
    parent_label: str | None = None,
    max_keywords: int = 2,
) -> tuple[str, list[str]]:
    """
    Deterministic label + keywords. Returns (name, keywords).
    The name is short and does not embed keywords.
    """
    if not texts:
        return (parent_label or "Misc", [])

    terms: list[str] = []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=500,
        )
        X = vectorizer.fit_transform(texts)
        if X.shape[1] != 0:
            scores = np.asarray(X.mean(axis=0)).ravel()
            top_idx = scores.argsort()[::-1][:max_keywords]
            terms = [vectorizer.get_feature_names_out()[i] for i in top_idx if scores[i] > 0]
    except Exception:
        terms = []

    name = parent_label or "Topic"
    if terms and not parent_label:
        name = " ".join(terms[:2])
    return (name, terms[:max_keywords])


def detect_near_duplicate_labels(
    labels: List[str],
    threshold: float = 0.9,
) -> List[Tuple[int, int, float]]:
    """
    Detect near-duplicate labels using TF-IDF cosine similarity.
    Returns list of (i, j, similarity) with i < j and similarity >= threshold.
    """
    if len(labels) < 2:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vec.fit_transform(labels)
        sim = cosine_similarity(X)
        dupes = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if sim[i, j] >= threshold:
                    dupes.append((i, j, float(sim[i, j])))
        return dupes
    except Exception:
        return []
