from __future__ import annotations

import numpy as np
from typing import Any

from em_scores import softmax_over_z_of_log_px_given_z


def compute_posterior_diagnostics(
    pzx: np.ndarray,
    choices_list: list[str],
    p_z_prior: np.ndarray,
    assigned_label: str | None = None,
) -> dict[str, Any]:
    """
    Compute diagnostic metrics from posterior p(z|x).

    Definitions:
      - pmax_posterior: max_j p(z_j | x)
      - pmax_lift: max softmax(log p(z|x) - log p(z)) == normalized p(z|x)/p(z)
      - assigned_pzx: p(z_assigned | x)
    """
    pzx = np.asarray(pzx, dtype=float)
    pzx = np.clip(pzx, 1e-30, 1.0)
    pzx = pzx / pzx.sum()

    top_order = np.argsort(-pzx)
    top1_idx = int(top_order[0])
    top2_idx = int(top_order[1]) if len(top_order) > 1 else None

    top1_pzx = float(pzx[top1_idx])
    top2_pzx = float(pzx[top2_idx]) if top2_idx is not None else float("nan")
    top_gap = float(top1_pzx - top2_pzx) if np.isfinite(top2_pzx) else float("nan")

    entropy = float(-np.sum(pzx * np.log(pzx)))

    label_to_idx = {str(lbl): i for i, lbl in enumerate(choices_list)}
    assigned_pzx = float("nan")
    assigned_rank = None
    is_assigned_top1 = None
    if assigned_label is not None:
        assigned_idx = label_to_idx.get(str(assigned_label))
        if assigned_idx is not None and assigned_idx < len(pzx):
            assigned_pzx = float(pzx[assigned_idx])
            assigned_rank = int(np.where(top_order == assigned_idx)[0][0]) + 1
            is_assigned_top1 = (assigned_rank == 1)

    log_pzx = np.log(np.clip(pzx, 1e-30, None))
    log_pz = np.log(np.clip(p_z_prior, 1e-30, None))
    pxz_rel = softmax_over_z_of_log_px_given_z(log_pzx - log_pz)
    pmax_lift = float(np.max(pxz_rel))
    lift_top1_idx = int(np.argmax(pxz_rel))
    lift_top1_label = None
    if lift_top1_idx != top1_idx:
        lift_top1_label = choices_list[lift_top1_idx]

    return {
        "pmax": top1_pzx,  # backward-compat
        "pmax_posterior": top1_pzx,
        "top1_label": choices_list[top1_idx],
        "top1_pzx": top1_pzx,
        "top2_label": choices_list[top2_idx] if top2_idx is not None else None,
        "top2_pzx": top2_pzx,
        "top1_top2_gap": top_gap,
        "entropy": entropy,
        "assigned_pzx": assigned_pzx,
        "assigned_rank": assigned_rank,
        "is_assigned_top1": is_assigned_top1,
        "pmax_lift": pmax_lift,
        "pxz_max": pmax_lift,  # backward-compat
        "lift_top1_label": lift_top1_label,
    }
