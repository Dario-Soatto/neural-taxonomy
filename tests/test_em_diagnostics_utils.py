import os
import sys
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from em_diagnostics_utils import compute_posterior_diagnostics
from em_scores import softmax_over_z_of_log_px_given_z


def test_posterior_diagnostics_basic():
    pzx = np.array([0.1, 0.7, 0.2], dtype=float)
    prior = np.array([0.2, 0.5, 0.3], dtype=float)
    choices = ["A", "B", "C"]

    diag = compute_posterior_diagnostics(
        pzx=pzx,
        choices_list=choices,
        p_z_prior=prior,
        assigned_label="C",
    )

    assert diag["top1_label"] == "B"
    assert np.isclose(diag["top1_pzx"], 0.7)
    assert diag["top2_label"] == "C"
    assert np.isclose(diag["top2_pzx"], 0.2)
    assert np.isclose(diag["top1_top2_gap"], 0.5)
    assert diag["assigned_rank"] == 2
    assert diag["is_assigned_top1"] is False

    # Entropy over posterior
    expected_entropy = -np.sum(pzx * np.log(pzx))
    assert np.isclose(diag["entropy"], expected_entropy)

    # Lift metric: softmax(log pzx - log prior)
    expected_lift = softmax_over_z_of_log_px_given_z(np.log(pzx) - np.log(prior)).max()
    assert np.isclose(diag["pmax_lift"], expected_lift)
