import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import pandas as pd

from schema_utils import (
    propose_label_from_texts,
    detect_near_duplicate_labels,
    sanitize_text,
    build_label_string,
)
from run_em_algorithm import _rename_duplicate_labels, _select_poor_candidates


def test_propose_label_from_texts_basic():
    texts = [
        "Hillary Clinton shepherded the Adoption and Safe Families Act.",
        "She helped pass the Foster Care Independence Act.",
    ]
    label = propose_label_from_texts(texts, parent_label="Policy")
    assert isinstance(label, str)
    assert len(label) > 0
    assert "Policy" in label


def test_detect_near_duplicate_labels():
    labels = [
        "Action: policy reform",
        "Action: policy reform",
        "Background: early life",
    ]
    dupes = detect_near_duplicate_labels(labels, threshold=0.95)
    assert any(i == 0 and j == 1 for i, j, _ in dupes)


def test_sanitize_and_cap_label():
    assert sanitize_text(float("nan")) == ""
    long_name = "A" * 200
    label = build_label_string(long_name, keywords=["one", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])
    assert len(label) <= 160


def test_rename_duplicate_labels_cluster_level():
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2, 3, 3],
            "agglomerative_label": ["A", "A", "B", "B", "B", "B"],
            "sentence_text": ["t1", "t2", "t3", "t4", "t5", "t6"],
        }
    )
    # Only clusters 2 and 3 are duplicates at cluster level ("B").
    changed = _rename_duplicate_labels(df, "sentence_text")
    assert changed > 0
    labels = df.groupby("cluster_id")["agglomerative_label"].first().tolist()
    assert len(set(labels)) == len(labels)


def test_rename_duplicate_labels_noop_when_unique():
    df = pd.DataFrame(
        {
            "cluster_id": [1, 1, 2, 2, 3, 3],
            "agglomerative_label": ["A", "A", "B", "B", "C", "C"],
            "sentence_text": ["t1", "t2", "t3", "t4", "t5", "t6"],
        }
    )
    changed = _rename_duplicate_labels(df, "sentence_text")
    assert changed == 0


def test_select_poor_candidates_bottomk():
    # pmax slightly above 0.2 but still bottom-K
    pmax = [0.21, 0.22, 0.23, 0.5, 0.6]
    entropy = [1.5, 1.4, 1.3, 0.2, 0.1]
    ranks = [2, 3, 2, 1, 1]
    idx = _select_poor_candidates(pmax, entropy, ranks, min_poorly_explained_for_add=2)
    assert len(idx) >= 2
