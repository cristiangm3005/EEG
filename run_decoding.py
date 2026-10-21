"""Decode gain vs loss feedback and explain what the models use.

    python run_decoding.py --n-subjects 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))
from eegxai.simulate import simulate_subject, EVENT_ID  # noqa: E402
from eegxai.decoding import (prepare_epochs, time_resolved, temporal_generalisation,  # noqa: E402
                             activation_patterns, above_chance_cluster)
from eegxai.explain import trial_features, explain  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)

    scores, gens, pats, filts, feat_tables, labels = [], [], [], [], [], []
    for i in range(1, args.n_subjects + 1):
        raw = simulate_subject(f"{i:02d}", n_trials=160, rewp_effect_uv=rng.normal(4, 1), seed=args.seed + i)
        ep = prepare_epochs(raw, EVENT_ID)
        scores.append(time_resolved(ep, seed=args.seed))
        gens.append(temporal_generalisation(ep, seed=args.seed))
        f, p = activation_patterns(ep, 0.25, 0.35)
        filts.append(f); pats.append(p)
        ft = trial_features(ep)
        feat_tables.append((ft - ft.mean()) / ft.std())  # within-subject z-scoring before pooling
        labels.append((ep.events[:, 2] == 1).astype(int))
        peak = ep.times[np.argmax(scores[-1])]
        print(f"sub-{i:02d}: peak AUC {scores[-1].max():.2f} at {peak * 1000:.0f} ms")

    times = ep.times
    S = np.array(scores)
    clusters = above_chance_cluster(S, seed=args.seed)
    sig = [c for c in clusters if c["p"] < 0.05]
    for c in sig:
        t = times[c["mask"]]
        print(f"Above-chance decoding: {t.min() * 1000:.0f}-{t.max() * 1000:.0f} ms (cluster p = {c['p']:.3f})")

    X = pd.concat(feat_tables, ignore_index=True)
    y = np.concatenate(labels)
    ex = explain(X, y, seed=args.seed)
    ex["permutation"].to_csv(out / "permutation_importance.csv", index=False)
    if ex["shap"] is not None:
        ex["shap"].to_csv(out / "shap_importance.csv", index=False)
    print(f"Feature-based classifier (pooled trials), held-out AUC = {ex['test_auc']:.2f}")
    print("Top features (SHAP):" if ex["shap"] is not None else "Top features (permutation):")
    top = (ex["shap"] if ex["shap"] is not None else ex["permutation"]).head(5)
    print(top.to_string(index=False))

    pd.DataFrame(S, columns=np.round(times, 3)).to_csv(out / "time_resolved_auc.csv", index=False)

    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, :2])
    m, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(len(S))
    ax.fill_between(times * 1000, m - se, m + se, color="#2166ac", alpha=0.25)
    ax.plot(times * 1000, m, color="#2166ac")
    for c in sig:
        ax.plot(times[c["mask"]] * 1000, np.full(c["mask"].sum(), 0.46), color="k", lw=3)
    ax.axhline(0.5, color="0.5", ls="--", lw=0.8); ax.axvline(0, color="0.5", lw=0.8)
    ax.set(xlabel="Time (ms)", ylabel="ROC-AUC", title="Time-resolved decoding of gain vs loss (mean ± SEM; bar: cluster p < .05)")

    ax = fig.add_subplot(gs[0, 2])
    G = np.mean(gens, axis=0)
    im = ax.imshow(G, origin="lower", cmap="RdBu_r", vmin=0.3, vmax=0.7,
                   extent=np.array([times[0], times[-1], times[0], times[-1]]) * 1000)
    ax.set(xlabel="Test time (ms)", ylabel="Train time (ms)", title="Temporal generalisation")
    plt.colorbar(im, ax=ax, fraction=0.046, label="AUC")

    info = ep.info
    for k, (arr, title) in enumerate([(np.mean(filts, 0), "Decoder weights (filters, normalised)\nnot interpretable as sources"),
                                      (np.mean(pats, 0), "Activation patterns (Haufe, gain > loss)\ninterpretable forward model")]):
        ax = fig.add_subplot(gs[1, k])
        mne.viz.plot_topomap(arr, info, axes=ax, show=False, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)

    ax = fig.add_subplot(gs[1, 2])
    tbl = (ex["shap"] if ex["shap"] is not None else ex["permutation"]).head(8).iloc[::-1]
    col = "mean_abs_shap" if ex["shap"] is not None else "importance"
    ax.barh(tbl["feature"], tbl[col], color="#4d4d4d")
    ax.set(xlabel="mean |SHAP|" if ex["shap"] is not None else "Permutation importance",
           title="Most informative trial features")
    ax.tick_params(axis="y", labelsize=7.5)
    fig.tight_layout(); fig.savefig(out / "decoding_xai.png", dpi=150)
    print(f"Outputs written to {out.resolve()}")


if __name__ == "__main__":
    main()
