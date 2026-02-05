# src/em_scores.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List

def _safe_log(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return np.log(np.clip(x, eps, None))

def bayes_px_given_z_normalized(p_z_given_x: np.ndarray, p_z_prior: np.ndarray) -> np.ndarray:
    """
    p(x|z) ∝ p(z|x) p(z), normalized across z by sum_k p(z_k|x) p(z_k).
    Returns a K-vector that sums to 1 for each x.
    """
    numer = p_z_given_x * p_z_prior
    denom = np.sum(numer)
    if denom <= 0:
        return np.ones_like(numer) / len(numer)
    return numer / denom

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
      - L_z[j]: average log p(x|z_j) using Bayes-normalized scores
      - C_z[j]: average posterior p(z_j|x)
      - Optional centroids/variances and pairwise centroid cosine similarities
      - Per-row pmax and argmax under p(x|z)
    """
    if p_z_prior is None:
        labels_as_str = df[cluster_col].astype(str)
        choices_str = [str(c) for c in choices]
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

    try:
        L_baseline = float(prob_calibrator.compute_average_log_px_from_sample(list(df[text_col].astype(str))))
    except AttributeError:
        try:
            L_baseline = float(np.mean([prob_calibrator.compute_log_px(x) for x in df[text_col].astype(str)]))
        except AttributeError:
            L_baseline = float(np.mean([np.log(max(prob_calibrator.compute_p_X(x), 1e-30)) for x in df[text_col].astype(str)]))

    posteriors = []
    for x in df[text_col].astype(str):
        pzx = prob_calibrator.calibrate_p_z_given_X(x)
        posteriors.append(pzx)
    PZX = np.vstack(posteriors)
    if PZX.shape[1] != len(choices):
        raise ValueError(
            f"PZX width ({PZX.shape[1]}) does not match number of choices ({len(choices)}). "
            "Calibrator choices are out of sync with current schema labels."
        )

    PX_given_Z_norm = np.vstack([bayes_px_given_z_normalized(PZX[i], pz) for i in range(PZX.shape[0])])
    log_PX_given_Z_norm = _safe_log(PX_given_Z_norm)

    pmax = PX_given_Z_norm.max(axis=1)
    z_hat = PX_given_Z_norm.argmax(axis=1)

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
        L_z[j] = log_PX_given_Z_norm[mask, j].mean() if mask.any() else float('-inf')
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
        "row_pmax": pmax,
        "row_z_hat": z_hat,
        "row_log_px_given_z_norm": log_PX_given_Z_norm,
        "row_px_given_z_norm": PX_given_Z_norm,
        "PZX": PZX,
        "pz_prior": pz,
        "row_cluster_idx": assigned_idx,
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
