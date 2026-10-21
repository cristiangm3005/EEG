"""Multivariate decoding of reward feedback with interpretable outputs.

* Time-resolved decoding: a classifier at each time point (cross-validated
  ROC-AUC), revealing *when* gain and loss trials become distinguishable.
* Temporal generalisation: train at time t, test at t', revealing whether the
  underlying neural code is sustained or evolves.
* Activation patterns (Haufe et al., 2014): the weights of a linear decoder are
  filters, not maps of where the signal comes from. They are transformed into
  forward-model patterns, which *can* be interpreted neurophysiologically.
"""
from __future__ import annotations

import numpy as np
import mne
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, LinearModel,
                          cross_val_multiscore, get_coef, Vectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def prepare_epochs(raw: mne.io.BaseRaw, event_id: dict, decim: int = 2) -> mne.Epochs:
    raw = raw.copy().load_data().filter(0.1, 30.0, verbose="error").set_eeg_reference("average", verbose="error")
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
    return mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), decim=decim,
                      reject=dict(eeg=200e-6), preload=True, verbose="error")


def _clf(C: float = 1.0):
    return make_pipeline(StandardScaler(), LinearModel(LogisticRegression(C=C, solver="liblinear")))


def time_resolved(epochs: mne.Epochs, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    X, y = epochs.get_data(), epochs.events[:, 2]
    est = SlidingEstimator(_clf(), scoring="roc_auc", n_jobs=1, verbose="error")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_multiscore(est, X, y, cv=cv, n_jobs=1, verbose="error").mean(axis=0)


def temporal_generalisation(epochs: mne.Epochs, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    X, y = epochs.get_data(), epochs.events[:, 2]
    est = GeneralizingEstimator(_clf(), scoring="roc_auc", n_jobs=1, verbose="error")
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    return cross_val_multiscore(est, X, y, cv=cv, n_jobs=1, verbose="error").mean(axis=0)


def activation_patterns(epochs: mne.Epochs, tmin: float, tmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit one decoder on the average signal in [tmin, tmax]; return (filters, patterns) per channel."""
    ep = epochs.copy().crop(tmin, tmax)
    X = ep.get_data().mean(axis=2) * 1e6          # µV, no scaling: patterns stay in sensor units
    y = (ep.events[:, 2] == 1).astype(int)          # positive class = gain
    model = LinearModel(LogisticRegression(C=0.1, solver="liblinear")).fit(X, y)
    filters = model.filters_.ravel()
    patterns = model.patterns_.ravel()              # A = Cov(X) w  (Haufe et al., 2014)
    return filters / np.abs(filters).max(), patterns / np.abs(patterns).max()


def above_chance_cluster(scores: np.ndarray, n_permutations: int = 2000, seed: int = 0) -> list[dict]:
    """Group-level one-sample cluster test of AUC - 0.5 over time (subjects x times)."""
    t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
        scores - 0.5, n_permutations=n_permutations, tail=1, seed=seed, out_type="mask", verbose="error")
    out = []
    for c, p in zip(clusters, pvals):
        mask = np.zeros(scores.shape[1], dtype=bool)
        mask[c] = True  # MNE returns a tuple of indices for 1-D data
        out.append({"mask": mask, "p": float(p)})
    return out
