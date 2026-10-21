"""Synthetic EEG generator for a gambling (reward feedback) task.

Produces an ``mne.io.RawArray`` with 32 standard 10-20 channels, realistic
1/f background activity, alpha rhythm, line noise, eye blinks, and a
feedback-locked Reward Positivity (RewP) that is larger after gains than
after losses over fronto-central sites.

The simulator exists so the full pipeline can run offline and be unit tested.
Copied from the companion repository eeg-reward-positivity so this one runs standalone.
"""
from __future__ import annotations

import numpy as np
import mne

CH_NAMES = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6",
    "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3",
    "Pz", "P4", "P8", "PO9", "O1", "Oz", "O2", "PO10", "FCz", "AFz",
]

EVENT_ID = {"feedback/gain": 1, "feedback/loss": 2}


def _pink_noise(n_times: int, rng: np.random.Generator, exponent: float = 1.0) -> np.ndarray:
    """1/f^exponent noise via spectral shaping."""
    white = rng.standard_normal(n_times)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_times)
    freqs[0] = freqs[1]
    spec /= freqs ** (exponent / 2.0)
    out = np.fft.irfft(spec, n=n_times)
    return out / out.std()


def _rewp_waveform(times: np.ndarray, amp: float, latency: float = 0.30, width: float = 0.05) -> np.ndarray:
    return amp * np.exp(-0.5 * ((times - latency) / width) ** 2)


def simulate_subject(
    subject: str = "01",
    n_trials: int = 120,
    sfreq: float = 250.0,
    rewp_effect_uv: float = 4.0,
    seed: int | None = 0,
    sensor_noise_uv: float = 10.0,
) -> mne.io.RawArray:
    """Simulate one participant performing a doors/gambling task.

    Parameters
    ----------
    subject : participant label, stored in ``raw.info['subject_info']``.
    n_trials : number of feedback events (balanced gain/loss).
    sfreq : sampling frequency in Hz.
    rewp_effect_uv : gain-minus-loss RewP amplitude at FCz (microvolts).
    seed : random seed for reproducibility.
    sensor_noise_uv : SD of spatially independent sensor noise (makes single-trial
        decoding realistically hard; spatially correlated noise alone is easy to filter out).
    """
    rng = np.random.default_rng(seed)
    n_ch = len(CH_NAMES)
    isi = rng.uniform(2.0, 2.6, size=n_trials)
    onsets = np.cumsum(isi) + 2.0
    duration = onsets[-1] + 3.0
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq

    # Background: spatially correlated pink noise + occipital alpha
    sources = np.stack([_pink_noise(n_times, rng) for _ in range(8)])
    mixing = rng.normal(0, 1, size=(n_ch, 8))
    data = mixing @ sources * 6e-6
    alpha = np.sin(2 * np.pi * rng.uniform(9.5, 10.5) * t) * 5e-6
    occ_idx = [CH_NAMES.index(c) for c in ("O1", "Oz", "O2", "P3", "Pz", "P4")]
    data[occ_idx] += alpha
    data += np.sin(2 * np.pi * 50.0 * t) * 1.5e-6  # line noise (EU, 50 Hz)
    data += rng.standard_normal(data.shape) * sensor_noise_uv * 1e-6  # independent sensor noise

    # Eye blinks: frontal-dominant
    blink_topo = np.array([1.0 if c.startswith(("Fp", "AF")) else 0.4 if c.startswith("F") else 0.05 for c in CH_NAMES])
    for b in rng.uniform(1, duration - 1, size=int(duration / 4)):
        bt = np.exp(-0.5 * ((t - b) / 0.08) ** 2) * 80e-6
        data += np.outer(blink_topo, bt)

    # Feedback events and RewP
    conditions = rng.permutation(np.repeat([1, 2], n_trials // 2))
    fcz_like = {"FCz": 1.0, "Fz": 0.8, "Cz": 0.8, "FC1": 0.75, "FC2": 0.75, "F3": 0.5, "F4": 0.5, "C3": 0.45, "C4": 0.45, "AFz": 0.4}
    topo = np.array([fcz_like.get(c, 0.1) for c in CH_NAMES])
    ep_t = np.arange(int(0.8 * sfreq)) / sfreq
    for onset, cond in zip(onsets, conditions):
        start = int(onset * sfreq)
        stop = start + ep_t.size
        if stop > n_times:
            break
        common = _rewp_waveform(ep_t, 3e-6, latency=0.22, width=0.06)  # generic feedback P2
        diff = _rewp_waveform(ep_t, rewp_effect_uv * 1e-6 * (0.5 if cond == 1 else -0.5))
        jitter = rng.normal(1.0, 0.25)
        data[:, start:stop] += np.outer(topo, (common + diff) * jitter)

    info = mne.create_info(CH_NAMES, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="error")
    montage_name = "colin27_1020" if "colin27_1020" in mne.channels.get_builtin_montages() else "standard_1020"
    raw.set_montage(montage_name, on_missing="ignore")
    raw.info["subject_info"] = {"his_id": f"sub-{subject}"}
    desc = ["feedback/gain" if c == 1 else "feedback/loss" for c in conditions]
    raw.set_annotations(mne.Annotations(onsets[: len(desc)], [0.0] * len(desc), desc))
    return raw
