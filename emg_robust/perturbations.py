"""The four EMG-Robust perturbation operators.

Each takes a batch of raw EMG (shape [batch, 16, window_length]) and returns
a perturbed copy. None of them depend on which model or dataset produced the
EMG -- see the paper's Method section for the exact equations these
implement, and `make_perturbation` below for how a sweep config maps to one
of these calls.
"""
from functools import partial

import numpy as np
import torch


def add_gaussian_noise(emg, level_frac, rng):
    """x_c[t] + alpha * sigma_c * N(0,1), where sigma_c is channel c's SD
    within this window and alpha = level_frac.

    `rng` must be a np.random.Generator created ONCE per (level, seed)
    config and reused across every batch in that run. Creating a fresh
    Generator on every call would give every window in the run the same
    noise draw instead of an independent one -- this was a real bug in an
    earlier version of this function; see the paper's reproducibility
    discussion.
    """
    std = emg.std(dim=-1, keepdim=True)
    noise = torch.from_numpy(rng.standard_normal(emg.shape).astype(np.float32))
    return emg + noise * std * level_frac


def select_dropout_channels(n_channels, seed, total_channels=16):
    """Pick `n_channels` of `total_channels` to drop, deterministically from
    (n_channels, seed) alone.

    Call this ONCE per seed, before the run starts. The selection then stays
    fixed for the whole evaluation, modeling a channel that is detached for
    an entire session rather than one re-sampled every window. Because the
    choice depends only on (n_channels, seed), the same seed always selects
    the same channels regardless of model or dataset -- at k=1 (mild
    dropout, 16 possible channels), two of the five seeds landing on the
    same channel is an expected ~48% chance, not a bug (see README).
    """
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_channels, size=n_channels, replace=False).tolist())


def apply_channel_mask(emg, channels):
    """Zero the given channel indices for every window in the batch."""
    emg = emg.clone()
    emg[:, channels, :] = 0.0
    return emg


def amplitude_scale(emg, scale):
    """x_c[t] * scale."""
    return emg * scale


def temporal_shift(emg, shift_samples):
    """Delay the signal by `shift_samples`, zero-padded at the start.

    A negative value would advance the signal instead; EMG-Robust only uses
    positive delays, modeling control-loop latency (the EMG arriving later
    than the model expects).
    """
    emg = emg.clone()
    if shift_samples == 0:
        return emg
    shifted = torch.zeros_like(emg)
    if shift_samples > 0:
        shifted[:, :, shift_samples:] = emg[:, :, :-shift_samples]
    else:
        shifted[:, :, :shift_samples] = emg[:, :, shift_samples:]
    return shifted


def make_perturbation(perturbation: str, value, seed=None):
    """Build the (perturb_fn, dropped_channels_str) pair for one sweep
    config, exactly as scripts/run_sweep.py's loop expects. `perturb_fn`
    takes a raw EMG batch and returns the perturbed batch; `perturb_fn` is
    None-safe to call with perturbation="baseline" by simply not calling
    this function at all (the sweep's baseline row skips perturbation).
    """
    if perturbation == "noise":
        rng = np.random.default_rng(seed)
        return partial(add_gaussian_noise, level_frac=value, rng=rng), ""
    if perturbation == "dropout":
        channels = select_dropout_channels(value, seed)
        return partial(apply_channel_mask, channels=channels), ",".join(str(c) for c in channels)
    if perturbation == "amplitude":
        return partial(amplitude_scale, scale=value), ""
    if perturbation == "temporal_shift":
        return partial(temporal_shift, shift_samples=value), ""
    raise ValueError(f"Unknown perturbation {perturbation!r}")
