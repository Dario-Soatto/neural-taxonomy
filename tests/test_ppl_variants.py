import os
import sys
import importlib.util
import numpy as np
import pytest


# Load src/em_scores.py directly to avoid importing src/__init__.py.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EM_SCORES_PATH = os.path.join(ROOT, "src", "em_scores.py")
spec = importlib.util.spec_from_file_location("em_scores", EM_SCORES_PATH)
em_scores = importlib.util.module_from_spec(spec)
sys.modules["em_scores"] = em_scores
spec.loader.exec_module(em_scores)

compute_corpus_level_scores_variants = em_scores.compute_corpus_level_scores_variants


def test_ppl_variants_oracle_soft_and_assigned_consistency():
    # N=3, K=3 synthetic conditional log-likelihoods.
    row_log_px_given_z = np.array(
        [
            [-2.0, -3.0, -4.0],
            [-1.0, -0.5, -2.0],
            [-3.0, -3.2, -2.5],
        ],
        dtype=float,
    )
    q_ij = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.3, 0.2, 0.5],
        ],
        dtype=float,
    )
    row_z_hat = np.argmax(row_log_px_given_z, axis=1).astype(int)
    row_cluster_idx = row_z_hat.copy()
    token_counts = np.array([5.0, 4.0, 6.0], dtype=float)

    # Per-row oracle ll upper-bounds soft ll.
    row_oracle = row_log_px_given_z[np.arange(row_log_px_given_z.shape[0]), row_z_hat]
    row_soft = np.sum(q_ij * row_log_px_given_z, axis=1)
    assert np.all(row_oracle >= row_soft - 1e-12)

    doc_scores = {
        "row_log_px_given_z": row_log_px_given_z,
        "row_z_hat": row_z_hat,
        "row_cluster_idx": row_cluster_idx,
        "PZX": q_ij,
        "pz_prior": np.array([0.3, 0.4, 0.3], dtype=float),
    }
    corpus = compute_corpus_level_scores_variants(
        doc_scores=doc_scores,
        token_counts=token_counts,
        k_complexity=3,
        is_test_mask=None,
    )

    # Oracle routing must not have worse perplexity than soft EM routing.
    assert corpus["perplexity_token_oracle"] <= corpus["perplexity_token_soft"] + 1e-10

    # If assigned indices equal z_hat for all rows, assigned and oracle coincide.
    assert np.isclose(
        corpus["avg_logL_per_token_oracle"],
        corpus["avg_logL_per_token_assigned"],
        atol=1e-12,
    )
    assert np.isclose(
        corpus["perplexity_token_oracle"],
        corpus["perplexity_token_assigned"],
        atol=1e-12,
    )


def test_assigned_variant_raises_on_missing_cluster_index():
    row_log_px_given_z = np.array([[-1.0, -2.0], [-0.5, -1.5]], dtype=float)
    doc_scores = {
        "row_log_px_given_z": row_log_px_given_z,
        "row_z_hat": np.array([0, 0], dtype=int),
        "row_cluster_idx": np.array([0, -1], dtype=int),
        "PZX": np.array([[0.9, 0.1], [0.7, 0.3]], dtype=float),
        "pz_prior": np.array([0.5, 0.5], dtype=float),
    }
    with pytest.raises(ValueError, match="missing label in choices"):
        compute_corpus_level_scores_variants(doc_scores=doc_scores, token_counts=np.array([2.0, 3.0]))
