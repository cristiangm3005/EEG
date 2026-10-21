import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from eegxai.simulate import simulate_subject, EVENT_ID  # noqa: E402
from eegxai.decoding import prepare_epochs, time_resolved, activation_patterns  # noqa: E402
from eegxai.explain import trial_features  # noqa: E402


def _epochs(effect=6.0, seed=0):
    return prepare_epochs(simulate_subject(n_trials=100, rewp_effect_uv=effect, seed=seed), EVENT_ID, decim=4)


def test_decoding_peaks_in_rewp_window():
    ep = _epochs()
    auc = time_resolved(ep, n_splits=4)
    peak_t = ep.times[np.argmax(auc)]
    assert auc.max() > 0.65 and 0.2 <= peak_t <= 0.4


def test_baseline_at_chance():
    ep = _epochs()
    auc = time_resolved(ep, n_splits=4)
    assert abs(auc[ep.times < 0].mean() - 0.5) < 0.1


def test_patterns_peak_frontocentral():
    ep = _epochs(effect=8.0, seed=1)
    _, pat = activation_patterns(ep, 0.25, 0.35)
    top = ep.ch_names[int(np.argmax(np.abs(pat)))]
    assert top in {"FCz", "Fz", "Cz", "FC1", "FC2", "AFz", "F3", "F4"}


def test_feature_table_shape():
    ep = _epochs()
    X = trial_features(ep)
    assert len(X) == len(ep) and X.notna().all().all()
