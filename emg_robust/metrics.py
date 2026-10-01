"""Landmark-distance evaluation, using emg2pose's own metric implementation
directly so our baseline numbers are comparable to the original paper's."""
import torch
from emg2pose.metrics import LandmarkDistances

_metric_fn = LandmarkDistances()


def evaluate(loader, model_adapter, perturb_fn=None):
    """Run `model_adapter` over every window in `loader`, optionally
    perturbing the raw EMG first with `perturb_fn(emg) -> emg`.

    Returns (landmark_mm, fingertip_mm), each averaged over windows and
    weighted by batch size (the final batch of a session is usually
    smaller than the rest).
    """
    total_landmark = 0.0
    total_fingertip = 0.0
    total_windows = 0
    with torch.no_grad():
        for sample in loader:
            emg = sample["emg"].float()
            if perturb_fn is not None:
                emg = perturb_fn(emg)
            batch = {
                "emg": emg,
                "joint_angles": sample["joint_angles"].float(),
                "no_ik_failure": sample["no_ik_failure"],
            }
            pred, target, mask = model_adapter.predict(batch)
            result = _metric_fn(pred, target, mask, stage="eval")
            n = emg.shape[0]
            total_landmark += result["eval_landmark_distance"].item() * n
            total_fingertip += result["eval_fingertip_distance"].item() * n
            total_windows += n
    return total_landmark / total_windows, total_fingertip / total_windows
