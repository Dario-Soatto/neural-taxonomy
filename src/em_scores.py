# src/em_scores.py
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)

def _safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _validate_token_counts(token_counts: np.ndarray | None, n_rows: int) -> np.ndarray:
    if token_counts is None:
        token_counts = np.ones(n_rows, dtype=float)
    token_counts = np.asarray(token_counts, dtype=float)
    if token_counts.shape[0] != n_rows:
        raise ValueError(
            f"token_counts length ({token_counts.shape[0]}) must match number of rows ({n_rows})."
        )
    return token_counts

def bayes_log_px_given_z(
    p_z_given_x: np.ndarray,
    log_px: float,
    p_z_prior: np.ndarray,
    eps: float = 1e-30,
) -> np.ndarray:
    """
    Compute log p(x|z) using Bayes' rule:
      log p(x|z) = log p(z|x) + log p(x) - log p(z)
    Returns a K-vector of log p(x|z). No normalization across z.
    """
    p_z_given_x = np.clip(p_z_given_x, eps, None)
    p_z_prior = np.clip(p_z_prior, eps, None)
    return np.log(p_z_given_x) + float(log_px) - np.log(p_z_prior)

def softmax_over_z_of_log_px_given_z(log_px_given_z: np.ndarray) -> np.ndarray:
    """
    Diagnostic-only: turn log p(x|z) into a normalized vector over z.
    This is NOT p(x|z); it is a relative support across z for a fixed x.
    """
    log_px_given_z = np.asarray(log_px_given_z, dtype=float)
    max_lp = np.max(log_px_given_z)
    exps = np.exp(log_px_given_z - max_lp)
    denom = np.sum(exps)
    if denom <= 0:
        return np.ones_like(exps) / len(exps)
    return exps / denom

def compute_document_level_scores(
    df: pd.DataFrame,
    text_col: str,
    cluster_col: str,
    choices: List[str],
    prob_calibrator,
    embeddings: np.ndarray | None = None,
    p_z_prior: np.ndarray | None = None,
) -> Dict:
    """
    Computes:
      - L_baseline: average log p(x)
      - L_z[j]: average log p(x|z_j) using Bayes' rule in log-space
      - C_z[j]: average posterior p(z_j|x)
      - Optional centroids/variances and pairwise centroid cosine similarities
      - Per-row pmax (posterior) and argmax under log p(x|z)
    """
    choices_str = [str(c) for c in choices]
    if len(set(choices_str)) != len(choices_str):
        raise ValueError(
            "Choices passed to compute_document_level_scores must be unique. "
            "Duplicate labels detected in schema choices."
        )

    if p_z_prior is None:
        labels_as_str = df[cluster_col].astype(str)
        counts = pd.Categorical(labels_as_str, categories=choices_str).value_counts()
        pz = counts.to_numpy(dtype=float)
        pz = pz / pz.sum() if pz.sum() > 0 else np.ones(len(choices_str)) / len(choices_str)
    else:
        pz = np.array(p_z_prior, dtype=float)
        if pz.ndim != 1:
            raise ValueError(f"p_z_prior must be 1D, got shape {pz.shape}.")
        if len(pz) != len(choices):
            raise ValueError(
                f"p_z_prior length ({len(pz)}) does not match number of choices ({len(choices)}). "
                "This usually means cluster choices changed but calibrator/prior was not refreshed."
            )

    if getattr(prob_calibrator, "full_logprob_fn", None) is None:
        raise ValueError(
            "ProbabilityCalibrator.full_logprob_fn is required to compute log p(x) "
            "for Bayes-consistent log p(x|z) scoring."
        )
    texts = df[text_col].astype(str).tolist()
    log_px_values = np.array([float(prob_calibrator.full_logprob_fn(x)) for x in texts], dtype=float)
    L_baseline = float(np.mean(log_px_values)) if len(log_px_values) else 0.0

    posteriors = []
    for x in texts:
        pzx = prob_calibrator.calibrate_p_z_given_X(x)
        posteriors.append(pzx)
    PZX = np.vstack(posteriors)
    if PZX.shape[1] != len(choices):
        raise ValueError(
            f"PZX width ({PZX.shape[1]}) does not match number of choices ({len(choices)}). "
            "Calibrator choices are out of sync with current schema labels."
        )

    log_px_given_z = np.vstack(
        [bayes_log_px_given_z(PZX[i], log_px_values[i], pz) for i in range(PZX.shape[0])]
    )

    pmax_posterior = PZX.max(axis=1)
    z_hat = log_px_given_z.argmax(axis=1)

    K = len(choices)
    L_z = np.zeros(K)
    C_z = np.zeros(K)
    choice_to_idx = {str(c): i for i, c in enumerate(choices)}
    assigned_idx = (
        df[cluster_col]
        .astype(str)
        .map(lambda x: choice_to_idx.get(x, -1))
        .to_numpy()
    )
    for j in range(K):
        mask = (assigned_idx == j)
        L_z[j] = log_px_given_z[mask, j].mean() if mask.any() else float('-inf')
        C_z[j] = PZX[:, j].mean()

    centroids = None
    variances = None
    pairwise_cos = None
    if embeddings is not None:
        centroids = []
        variances = []
        for j in range(K):
            idx = np.where(z_hat == j)[0]
            if len(idx) == 0:
                centroids.append(np.zeros(embeddings.shape[1]))
                variances.append(0.0)
            else:
                E = embeddings[idx]
                mu = E.mean(axis=0)
                centroids.append(mu)
                variances.append(((E - mu) ** 2).sum(axis=1).mean())
        centroids = np.vstack(centroids)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
        normed = centroids / norms
        pairwise_cos = normed @ normed.T

    return {
        "L_baseline": float(L_baseline),
        "L_z": L_z,
        "C_z": C_z,
        "centroids": centroids,
        "variances": variances,
        "pairwise_cos": pairwise_cos,
        "row_pmax_posterior": pmax_posterior,
        "row_z_hat": z_hat,
        "row_log_px_given_z": log_px_given_z,
        "PZX": PZX,
        "pz_prior": pz,
        "row_cluster_idx": assigned_idx,
        "row_log_px": log_px_values,
    }

def compute_corpus_level_scores(
    log_px_given_z_hat: np.ndarray,
    token_counts: np.ndarray | None = None,
    k_complexity: int | None = None,
    is_test_mask: np.ndarray | None = None,
    q_ij: np.ndarray | None = None,
    log_px_given_z: np.ndarray | None = None,
    log_pz: np.ndarray | None = None,
) -> Dict:
    """
    Inputs:
      - log_px_given_z_hat: per-row log p(x | z_hat)
      - token_counts: per-row token count; if None, uniform counts are used
      - k_complexity: used for AIC/BIC
      - is_test_mask: mask for test rows; if None, all rows considered test
    """
    N = len(log_px_given_z_hat)
    if is_test_mask is None:
        is_test_mask = np.ones(N, dtype=bool)

    logL = float(np.sum(log_px_given_z_hat))

    test_logL = float(np.sum(log_px_given_z_hat[is_test_mask]))
    n_test = int(np.sum(is_test_mask))
    aic = bic = None
    if k_complexity is not None and n_test > 0:
        aic = 2 * k_complexity - 2 * test_logL
        bic = k_complexity * np.log(n_test) - 2 * test_logL

    token_counts = _validate_token_counts(token_counts, N)
    T = float(np.sum(token_counts))
    avg_logL_per_token = float(logL / max(T, 1.0))
    ppl = float(np.exp(-avg_logL_per_token))

    elbo = None
    if q_ij is not None and log_px_given_z is not None and log_pz is not None:
        q_ij = np.asarray(q_ij, dtype=float)
        log_px_given_z = np.asarray(log_px_given_z, dtype=float)
        log_pz = np.asarray(log_pz, dtype=float)
        if q_ij.ndim != 2 or log_px_given_z.shape != q_ij.shape or log_pz.ndim != 1:
            raise ValueError("ELBO inputs must have shapes: q_ij (N,K), log_px_given_z (N,K), log_pz (K,).")
        log_q = _safe_log(q_ij)
        elbo = float(np.sum(q_ij * (log_px_given_z + log_pz[None, :] - log_q)))

    return {
        "logL_cond_total": logL,
        "token_count_total": T,
        "avg_logL_per_token": avg_logL_per_token,
        "AIC": aic,
        "BIC": bic,
        "perplexity": ppl,
        "perplexity_token": ppl,
        "ELBO": elbo,
    }


def compute_corpus_level_scores_variants(
    doc_scores: Dict,
    token_counts: np.ndarray | None = None,
    k_complexity: int | None = None,
    is_test_mask: np.ndarray | None = None,
) -> Dict:
    """
    Compute corpus-level conditional likelihood/perplexity under three routing variants:
      - oracle: z_hat = argmax_z log p(x|z)
      - assigned: currently assigned cluster index for each row
      - soft: expectation under q_ij = P(z|x)

    Existing keys are preserved and mapped to oracle for backward compatibility.
    """
    row_log_px_given_z = np.asarray(doc_scores["row_log_px_given_z"], dtype=float)
    row_z_hat = np.asarray(doc_scores["row_z_hat"], dtype=int)
    row_cluster_idx = np.asarray(doc_scores["row_cluster_idx"], dtype=int)
    q_ij = np.asarray(doc_scores["PZX"], dtype=float)

    if row_log_px_given_z.ndim != 2:
        raise ValueError(f"row_log_px_given_z must be 2D (N,K), got shape {row_log_px_given_z.shape}.")
    n_rows, n_choices = row_log_px_given_z.shape
    if row_z_hat.shape[0] != n_rows:
        raise ValueError(f"row_z_hat length ({row_z_hat.shape[0]}) must equal N ({n_rows}).")
    if row_cluster_idx.shape[0] != n_rows:
        raise ValueError(f"row_cluster_idx length ({row_cluster_idx.shape[0]}) must equal N ({n_rows}).")
    if q_ij.shape != row_log_px_given_z.shape:
        raise ValueError(
            f"PZX shape ({q_ij.shape}) must match row_log_px_given_z shape ({row_log_px_given_z.shape})."
        )

    if np.any(row_cluster_idx < 0):
        missing_rows = np.where(row_cluster_idx < 0)[0][:10].tolist()
        raise ValueError(
            "Found row_cluster_idx == -1 (missing label in choices; likely calibrator/schema mismatch). "
            f"Example row indices: {missing_rows}"
        )

    token_counts = _validate_token_counts(token_counts, n_rows)
    if is_test_mask is None:
        is_test_mask = np.ones(n_rows, dtype=bool)
    else:
        is_test_mask = np.asarray(is_test_mask, dtype=bool)
        if is_test_mask.shape[0] != n_rows:
            raise ValueError(f"is_test_mask length ({is_test_mask.shape[0]}) must equal N ({n_rows}).")

    row_ids = np.arange(n_rows)
    row_ll_oracle = row_log_px_given_z[row_ids, row_z_hat]
    row_ll_assigned = row_log_px_given_z[row_ids, row_cluster_idx]
    row_ll_soft = np.sum(q_ij * row_log_px_given_z, axis=1)

    # Oracle must upper-bound soft row-wise: max_z a_z >= E_q[a_z].
    eps = 1e-8
    row_diff = row_ll_soft - row_ll_oracle
    max_violation = float(np.max(row_diff)) if row_diff.size else 0.0
    if max_violation > eps:
        max_row = int(np.argmax(row_diff))
        logger.warning(
            "Invariant warning: oracle row ll < soft row ll by %.6e at row %s "
            "(oracle=%.6f, soft=%.6f, assigned_idx=%s, z_hat=%s).",
            max_violation,
            max_row,
            float(row_ll_oracle[max_row]),
            float(row_ll_soft[max_row]),
            int(row_cluster_idx[max_row]),
            int(row_z_hat[max_row]),
        )

    def _aggregate(row_ll: np.ndarray, suffix: str) -> dict:
        logl_total = float(np.sum(row_ll))
        token_total = float(np.sum(token_counts))
        avg_logl_tok = float(logl_total / max(token_total, 1.0))
        ppl_tok = float(np.exp(-avg_logl_tok))
        out = {
            f"logL_total_{suffix}": logl_total,
            f"avg_logL_per_token_{suffix}": avg_logl_tok,
            f"perplexity_token_{suffix}": ppl_tok,
            f"perplexity_{suffix}": ppl_tok,
        }
        if k_complexity is not None:
            test_logl = float(np.sum(row_ll[is_test_mask]))
            n_test = int(np.sum(is_test_mask))
            if n_test > 0:
                out[f"AIC_{suffix}"] = float(2 * k_complexity - 2 * test_logl)
                out[f"BIC_{suffix}"] = float(k_complexity * np.log(n_test) - 2 * test_logl)
            else:
                out[f"AIC_{suffix}"] = None
                out[f"BIC_{suffix}"] = None
        return out

    metrics = {
        "token_count_total": float(np.sum(token_counts)),
    }
    metrics.update(_aggregate(row_ll_oracle, "oracle"))
    metrics.update(_aggregate(row_ll_assigned, "assigned"))
    metrics.update(_aggregate(row_ll_soft, "soft"))

    if metrics["perplexity_token_oracle"] > metrics["perplexity_token_soft"] + 1e-8:
        max_row = int(np.argmax(row_diff)) if row_diff.size else -1
        logger.warning(
            "Invariant warning: PPL_tok_oracle (%.6f) > PPL_tok_soft (%.6f). "
            "Max row soft-oracle diff at row %s is %.6e.",
            float(metrics["perplexity_token_oracle"]),
            float(metrics["perplexity_token_soft"]),
            max_row,
            max_violation,
        )

    log_pz = np.log(np.clip(np.asarray(doc_scores["pz_prior"], dtype=float), 1e-30, None))
    log_q = _safe_log(q_ij)
    elbo = float(np.sum(q_ij * (row_log_px_given_z + log_pz[None, :] - log_q)))
    metrics["ELBO"] = elbo

    # Backward-compatible aliases map to oracle variant.
    metrics["logL_cond_total"] = metrics["logL_total_oracle"]
    metrics["avg_logL_per_token"] = metrics["avg_logL_per_token_oracle"]
    metrics["perplexity"] = metrics["perplexity_oracle"]
    metrics["perplexity_token"] = metrics["perplexity_token_oracle"]
    metrics["AIC"] = metrics.get("AIC_oracle")
    metrics["BIC"] = metrics.get("BIC_oracle")

    return metrics
