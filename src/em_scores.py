# src/em_scores.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def _safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

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

    if token_counts is None:
        token_counts = np.ones(N)
    T = float(np.sum(token_counts))
    ppl = float(np.exp(- np.sum(log_px_given_z_hat) / max(T, 1.0)))

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
        "AIC": aic,
        "BIC": bic,
        "perplexity": ppl,
        "ELBO": elbo,
    }
