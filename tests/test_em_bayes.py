import os
import sys
import importlib.util
import numpy as np
import pandas as pd

# Load src/em_scores.py directly to avoid importing src/__init__.py (which pulls heavy deps).
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EM_SCORES_PATH = os.path.join(ROOT, "src", "em_scores.py")
spec = importlib.util.spec_from_file_location("em_scores", EM_SCORES_PATH)
em_scores = importlib.util.module_from_spec(spec)
sys.modules["em_scores"] = em_scores
spec.loader.exec_module(em_scores)

bayes_log_px_given_z = em_scores.bayes_log_px_given_z
compute_document_level_scores = em_scores.compute_document_level_scores


def test_bayes_log_px_given_z_basic():
    pzx = np.array([0.2, 0.5, 0.3])
    pz = np.array([0.1, 0.7, 0.2])
    log_px = -10.0
    out = bayes_log_px_given_z(pzx, log_px, pz)
    expected = np.log(pzx) + log_px - np.log(pz)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, expected, atol=1e-9)


class _StubCalibrator:
    def __init__(self, pzx_map, log_px_map):
        self._pzx_map = pzx_map
        self._log_px_map = log_px_map
        self.full_logprob_fn = lambda x: self._log_px_map[x]

    def calibrate_p_z_given_X(self, x):
        return self._pzx_map[x]


def test_compute_document_level_scores_z_hat():
    choices = ["A", "B", "C"]
    df = pd.DataFrame(
        {
            "text": ["x0", "x1"],
            "label": ["A", "B"],
        }
    )
    pzx_map = {
        "x0": np.array([0.7, 0.2, 0.1]),
        "x1": np.array([0.1, 0.6, 0.3]),
    }
    log_px_map = {"x0": -5.0, "x1": -7.0}
    pz_prior = np.array([0.2, 0.5, 0.3])
    cal = _StubCalibrator(pzx_map, log_px_map)

    scores = compute_document_level_scores(
        df=df,
        text_col="text",
        cluster_col="label",
        choices=choices,
        prob_calibrator=cal,
        embeddings=None,
        p_z_prior=pz_prior,
    )

    # z_hat should follow argmax(log p(z|x) - log p(z)) since log p(x) is constant per row
    expected_z_hat = []
    for x in df["text"]:
        pzx = pzx_map[x]
        expected_z_hat.append(int(np.argmax(np.log(pzx) - np.log(pz_prior))))

    assert scores["row_z_hat"].tolist() == expected_z_hat
