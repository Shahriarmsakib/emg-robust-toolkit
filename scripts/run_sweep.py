#!/usr/bin/env python3
"""Run the EMG-Robust perturbation sweep for one baseline model against one
held-out emg2pose user session, writing results/final_<model>_user<N>.csv.

    python scripts/run_sweep.py --model neuropose --user 1
    python scripts/run_sweep.py --model vemg2pose --user 2

Requires emg2pose and UmeTrack installed (see README "Run a sweep" for the
one-time setup) and PyTorch (CPU is fine; a full sweep is ~37 forward passes
over 406 windows). Checkpoints and the two held-out sessions are downloaded
automatically into --cache on first use.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # running `python scripts/run_sweep.py` puts scripts/ (not
                                # the repo root) on sys.path, so emg_robust won't import
                                # without this -- it's never pip-installed anywhere in setup.

from emg_robust import adapter, data, sweep  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(adapter.CHECKPOINT_NAME),
                     help="which baseline to evaluate")
    ap.add_argument("--user", required=True, type=int, choices=sorted(data.USER_SESSION),
                     help="which held-out user session")
    ap.add_argument("--out", type=Path, default=None,
                     help="output CSV (default: results/final_<model>_user<N>.csv)")
    ap.add_argument("--cache", type=Path, default=REPO / ".cache",
                     help="where downloaded data/checkpoints are cached")
    a = ap.parse_args()

    out_path = a.out or REPO / "results" / f"final_{a.model}_user{a.user}.csv"
    ckpt_dir = data.fetch_checkpoints(a.cache)
    session_dir = data.get_session_dir(a.user, a.cache)

    model_adapter = adapter.load_model(a.model, ckpt_dir)
    loader = sweep.build_loader(session_dir)
    print(f"{len(loader.dataset)} windows total")

    sweep.run_sweep(model_adapter, loader, a.model, data.USER_SESSION[a.user], out_path)


if __name__ == "__main__":
    main()
