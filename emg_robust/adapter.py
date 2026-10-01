"""Thin wrapper around emg2pose's Emg2PoseModule, so the sweep and metrics
code never need to know which baseline checkpoint is loaded.

To point EMG-Robust at a different EMG->pose model, write your own loader
that returns an object with a `.predict(batch) -> (pred, target, mask)`
method with the same shapes as ModelAdapter.predict below, and pass it to
scripts/run_sweep.py (or call emg_robust.sweep.run_sweep directly) instead
of load_model(). The perturbations, metrics and report code only depend on
that one method, never on the model's internals.
"""
from pathlib import Path

import torch
from emg2pose.lightning import Emg2PoseModule

# Both are emg2pose's own released checkpoints, regression configuration
# (no ground-truth initial pose is given to the model).
CHECKPOINT_NAME = {
    "neuropose": "regression_neuropose.ckpt",
    "vemg2pose": "regression_vemg2pose.ckpt",
}


class ModelAdapter:
    """Wraps a loaded Emg2PoseModule for use in metrics.evaluate()."""

    def __init__(self, module: Emg2PoseModule):
        self.module = module
        self.module.eval()

    def predict(self, batch: dict):
        """batch: {"emg", "joint_angles", "no_ik_failure"}, as produced by
        emg2pose.data.WindowedEmgDataset. Returns (pred, target, mask)
        exactly as module.model(...) does -- metrics.evaluate() feeds this
        straight into emg2pose's own LandmarkDistances metric."""
        with torch.no_grad():
            return self.module.model(batch, provide_initial_pos=False)


def load_model(name: str, ckpt_dir: Path) -> ModelAdapter:
    """Load one of emg2pose's released regression-configuration baselines
    ("neuropose" or "vemg2pose") from its official checkpoint in ckpt_dir."""
    if name not in CHECKPOINT_NAME:
        raise ValueError(f"Unknown model {name!r}; known models: {sorted(CHECKPOINT_NAME)}")
    ckpt_path = Path(ckpt_dir) / CHECKPOINT_NAME[name]
    module = Emg2PoseModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,  # trusted official Meta checkpoint
    )
    return ModelAdapter(module)
