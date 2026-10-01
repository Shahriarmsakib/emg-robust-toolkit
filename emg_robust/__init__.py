"""EMG-Robust: a lightweight stress-testing toolkit for EMG-based pose models.

See the top-level README for usage. The modules here are:
    perturbations -- the four input perturbation operators
    adapter       -- loads an EMG->pose model (NeuroPose, vemg2pose, or your own)
    metrics       -- landmark-distance evaluation (wraps emg2pose.metrics)
    data          -- fetches the emg2pose checkpoints and the two held-out sessions
    sweep         -- builds the 37-config sweep and runs it, writing results/*.csv
"""
