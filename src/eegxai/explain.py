"""Feature-level explanations for a non-linear trial classifier.

Trials are summarised into interpretable features (mean amplitude per ROI in
consecutive 50 ms windows, plus theta and alpha band power per ROI). A
gradient-boosting classifier is trained on them and explained two ways:

* Permutation importance on held-out data (model-agnostic, global).
* SHAP values (TreeExplainer): per-trial attributions that can be aggregated
  per feature family, ROI or time window.

Interpretation caveat: correlated features share importance. Explanations
describe the model, not the brain, and should be checked against the
activation patterns of the linear decoder and against prior literature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import mne
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROIS = {"frontal": ["Fz", "F3", "F4", "AFz"], "frontocentral": ["FCz", "FC1", "FC2", "Cz"],
        "parietal": ["Pz", "P3", "P4", "CP1", "CP2"], "occipital": ["O1", "Oz", "O2"]}


def trial_features(epochs: mne.Epochs, win: float = 0.05) -> pd.DataFrame:
    data = epochs.get_data() * 1e6
    times = epochs.times
    idx = {c: i for i, c in enumerate(epochs.ch_names)}
    feats = {}
    starts = np.arange(0.0, 0.6, win)
    for roi, chs in ROIS.items():
        sel = [idx[c] for c in chs if c in idx]
        roi_sig = data[:, sel, :].mean(axis=1)
        for s in starts:
            m = (times >= s) & (times < s + win)
            feats[f"amp_{roi}_{int(s * 1000)}-{int((s + win) * 1000)}ms"] = roi_sig[:, m].mean(axis=1)
    psd = epochs.compute_psd(fmin=3, fmax=14, tmin=0.0, tmax=0.8, verbose="error")
    p, f = psd.get_data(), psd.freqs
    for roi, chs in ROIS.items():
        sel = [idx[c] for c in chs if c in idx]
        for band, (lo, hi) in {"theta": (4, 8), "alpha": (8, 13)}.items():
            fm = (f >= lo) & (f < hi)
            feats[f"pow_{roi}_{band}"] = np.log10(p[:, sel][:, :, fm].mean(axis=(1, 2)) + 1e-30)
    return pd.DataFrame(feats)


def explain(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> dict:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    clf = HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150, random_state=seed)
    clf.fit(Xtr, ytr)
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
    pi = permutation_importance(clf, Xte, yte, scoring="roc_auc", n_repeats=20, random_state=seed)
    perm = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean, "sd": pi.importances_std}) \
        .sort_values("importance", ascending=False)

    shap_df = None
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xte)
        sv = sv[1] if isinstance(sv, list) else sv
        shap_df = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)}) \
            .sort_values("mean_abs_shap", ascending=False)
    except Exception as exc:  # shap is optional
        print(f"SHAP skipped: {exc}")
    return {"test_auc": float(auc), "permutation": perm, "shap": shap_df}
