#!/usr/bin/env bash
# One-time setup for running a real sweep (scripts/run_sweep.py): clones
# emg2pose and its UmeTrack submodule (neither is on PyPI), applies the
# small Python-3.13 compatibility patch the repo needs (it was written for
# 3.10; a type-hint style that needs deferred annotation evaluation), and
# installs both plus this repo's own run_sweep.py dependencies.
#
# Does NOT download the dataset or checkpoints -- those are fetched
# automatically into --cache on first use by emg_robust/data.py, so you only
# need to run this script once before your first `python scripts/run_sweep.py`.
#
#   bash scripts/setup_emg2pose.sh
set -euo pipefail

REPO_DIR="emg2pose_repo"

if [ -d "$REPO_DIR" ]; then
    echo "$REPO_DIR already exists, skipping clone."
else
    git clone https://github.com/facebookresearch/emg2pose.git "$REPO_DIR"
fi

python3 -c "
from pathlib import Path
p = Path('$REPO_DIR/emg2pose/kinematics.py')
src = p.read_text()
if 'from __future__ import annotations' not in src:
    p.write_text('from __future__ import annotations\n' + src)
    print('Patched', p)
else:
    print(p, 'already patched.')
"

pip install -e "$REPO_DIR" -e "$REPO_DIR/emg2pose/UmeTrack"
pip install -r requirements.txt

echo "Done. Run, e.g.: python scripts/run_sweep.py --model neuropose --user 1"
