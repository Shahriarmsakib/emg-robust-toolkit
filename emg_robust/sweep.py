"""Builds the 37-config sweep (1 clean baseline + 36 perturbed) and runs it
against a loaded model, writing one row per config to a CSV -- the same
schema used for every result shipped with the paper.
"""
from pathlib import Path

import pandas as pd
from emg2pose.data import WindowedEmgDataset
from torch.utils.data import ConcatDataset, DataLoader

from . import metrics, perturbations

WINDOW_LENGTH = 10_000  # 5s at 2kHz
STRIDE = 10_000  # non-overlapping windows
BATCH_SIZE = 16
N_SEEDS = 5

NOISE_LEVELS = [("mild", 0.1), ("moderate", 0.3), ("severe", 0.6)]
DROPOUT_LEVELS = [("mild", 1), ("moderate", 2), ("severe", 4)]
AMPLITUDE_LEVELS = [("mild", 0.75), ("moderate", 0.5), ("severe", 0.25)]
SHIFT_LEVELS = [("mild", 40), ("moderate", 100), ("severe", 200)]  # samples @ 2kHz = 20/50/100ms


def build_configs():
    """36 perturbed configs, in the fixed order used throughout the paper:
    noise (3 levels x 5 seeds), dropout (3 x 5), amplitude (3, deterministic),
    temporal shift (3, deterministic). The sweep adds one more row for the
    unperturbed baseline, for 37 total."""
    configs = []
    for level, alpha in NOISE_LEVELS:
        for seed in range(N_SEEDS):
            configs.append(dict(perturbation="noise", level=level, value=alpha, seed=seed))
    for level, k in DROPOUT_LEVELS:
        for seed in range(N_SEEDS):
            configs.append(dict(perturbation="dropout", level=level, value=k, seed=seed))
    for level, a in AMPLITUDE_LEVELS:
        configs.append(dict(perturbation="amplitude", level=level, value=a, seed=None))
    for level, shift in SHIFT_LEVELS:
        configs.append(dict(perturbation="temporal_shift", level=level, value=shift, seed=None))
    return configs


def build_loader(session_dir: Path, batch_size: int = BATCH_SIZE) -> DataLoader:
    """Windows every .hdf5 file in session_dir into non-overlapping 5-second
    segments and concatenates them into one DataLoader (406 windows for the
    mini subset)."""
    files = sorted(Path(session_dir).glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files in {session_dir}")
    datasets = [WindowedEmgDataset(hdf5_path=f, window_length=WINDOW_LENGTH, stride=STRIDE) for f in files]
    return DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=False)


def run_sweep(model_adapter, loader, model_name: str, user_session: str, out_path: Path) -> Path:
    """Evaluate `model_adapter` on `loader` under every config, writing the
    growing result set to `out_path` after each config (so a crash partway
    through still leaves a usable, if partial, CSV)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    landmark, fingertip = metrics.evaluate(loader, model_adapter, perturb_fn=None)
    results.append(dict(model=model_name, user_session=user_session, perturbation="baseline", level="none",
                         value=None, seed=None, landmark_mm=landmark, fingertip_mm=fingertip, dropped_channels=""))
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"baseline: landmark={landmark:.4f} fingertip={fingertip:.4f}")

    for cfg in build_configs():
        perturb_fn, dropped = perturbations.make_perturbation(cfg["perturbation"], cfg["value"], cfg["seed"])
        landmark, fingertip = metrics.evaluate(loader, model_adapter, perturb_fn=perturb_fn)
        results.append(dict(model=model_name, user_session=user_session, perturbation=cfg["perturbation"],
                             level=cfg["level"], value=cfg["value"], seed=cfg["seed"],
                             landmark_mm=landmark, fingertip_mm=fingertip, dropped_channels=dropped))
        pd.DataFrame(results).to_csv(out_path, index=False)
        tag = f" channels={dropped}" if dropped else ""
        print(f"{cfg['perturbation']:15s} {cfg['level']:10s} seed={cfg['seed']}: "
              f"landmark={landmark:.4f} fingertip={fingertip:.4f}{tag}")

    print("Done:", out_path)
    return out_path
