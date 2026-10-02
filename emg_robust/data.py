"""Fetches the emg2pose checkpoints and the two held-out user sessions used
in the paper, caching everything under a local directory so repeat runs
don't re-download.

User 1 is emg2pose's official mini subset (~600MB, one held-out session).
User 2 is a second held-out session, pulled by streaming the full 431GB
archive and extracting only that session's files, stopping once they're all
found (or a safety cap is hit) rather than downloading the whole archive.

This mirrors the two notebooks' download/extraction cells, translated from
Colab's `!curl` + `!tar` shell commands into plain Python (requests +
tarfile) so it runs the same way as a script. The streaming-extraction logic
itself (CountingReader, the per-member session matching) is unchanged.
"""
import tarfile
from pathlib import Path

import pandas as pd
import requests

MINI_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
FULL_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
CKPT_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"

# Despite the name, these are per-user IDs (the `user` column in metadata.csv),
# not emg2pose session IDs (e.g. EXISTING_SESSION below) -- a single user's
# recordings span multiple sessions, and U1/U2 here are the same two users
# throughout the sweep regardless of which session a given recording came from.
USER_SESSION = {1: "d387095792", 2: "29ddab35d7"}
EXISTING_SESSION = "2022-12-06-1670313600-e3096-cv-emg-pose-train@2"  # the session already in the mini set
SAFETY_CAP_BYTES = 20 * 1024**3  # stop streaming the full archive after this many bytes scanned


def _download(url: str, dest: Path) -> None:
    resp = requests.get(url, stream=True)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)


def fetch_checkpoints(cache_dir: Path) -> Path:
    """Download + extract the official regression_* checkpoints, once."""
    ckpt_dir = Path(cache_dir) / "emg2pose_model_checkpoints"
    if ckpt_dir.exists():
        return ckpt_dir
    archive = Path(cache_dir) / "emg2pose_model_checkpoints.tar.gz"
    _download(CKPT_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir)
    return ckpt_dir


def fetch_mini_dataset(cache_dir: Path) -> Path:
    """Download + extract the official mini subset (User 1's session, plus
    metadata.csv for every session), once."""
    mini_dir = Path(cache_dir) / "emg2pose_dataset_mini"
    if mini_dir.exists():
        return mini_dir
    archive = Path(cache_dir) / "emg2pose_dataset_mini.tar"
    _download(MINI_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(cache_dir)
    return mini_dir


class _CountingReader:
    """Wraps a streamed HTTP response so tarfile can read it sequentially,
    without buffering the full (431GB) archive to disk or memory."""

    def __init__(self, resp):
        self.iterator = resp.iter_content(chunk_size=1024 * 1024)
        self.buffer = b""
        self.total_bytes = 0

    def read(self, size=-1):
        while size < 0 or len(self.buffer) < size:
            try:
                chunk = next(self.iterator)
            except StopIteration:
                break
            self.buffer += chunk
            self.total_bytes += len(chunk)
        if size < 0:
            data, self.buffer = self.buffer, b""
        else:
            data, self.buffer = self.buffer[:size], self.buffer[size:]
        return data


def fetch_second_session(cache_dir: Path, mini_dir: Path) -> Path:
    """Stream the full emg2pose archive and pull out the second held-out
    session (User 2), stopping once every file for that session is found or
    the safety cap is hit. Needs mini_dir's metadata.csv to identify which
    tar member belongs to which session."""
    out_dir = Path(cache_dir) / "emg2pose_second_session"
    if out_dir.exists() and any(out_dir.glob("*.hdf5")):
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(Path(mini_dir) / "metadata.csv")
    filename_to_session = dict(zip(meta["filename"], meta["session"]))
    filename_to_heldout = dict(zip(meta["filename"], meta["held_out_user"]))

    resp = requests.get(FULL_URL, stream=True)
    reader = _CountingReader(resp)
    found_session, saved_files = None, []

    with tarfile.open(fileobj=reader, mode="r|") as tar:
        for member in tar:
            if reader.total_bytes > SAFETY_CAP_BYTES:
                print(f"Hit safety cap ({reader.total_bytes / 1e9:.1f}GB scanned), "
                      f"stopping with {len(saved_files)} files.")
                break

            stem = Path(member.name).stem  # strips .hdf5, matches metadata's filename column
            session = filename_to_session.get(stem)
            if session is None or session == EXISTING_SESSION:
                continue
            if not filename_to_heldout.get(stem, False):
                continue

            if found_session is None:
                found_session = session
                print(f"Locking onto new held-out session: {session}")
            if session != found_session:
                continue

            f = tar.extractfile(member)
            if f is not None:
                fname = Path(member.name).name
                with open(out_dir / fname, "wb") as out:
                    out.write(f.read())
                saved_files.append(stem)
                print(f"  saved {fname} ({reader.total_bytes / 1e9:.2f}GB scanned so far)")

            expected = set(meta.loc[meta.session == found_session, "filename"])
            if expected.issubset(set(saved_files)):
                print(f"Got all {len(expected)} files for session {found_session} "
                      f"after {reader.total_bytes / 1e9:.1f}GB scanned")
                break

    print(f"Done. Scanned {reader.total_bytes / 1e9:.2f}GB, "
          f"saved {len(saved_files)} files for session {found_session}")
    return out_dir


def get_session_dir(user: int, cache_dir: Path) -> Path:
    """Return the directory of .hdf5 files for held-out User 1 or User 2,
    downloading/extracting/streaming whatever is missing from cache_dir."""
    cache_dir = Path(cache_dir)
    mini_dir = fetch_mini_dataset(cache_dir)
    if user == 1:
        return mini_dir
    if user == 2:
        return fetch_second_session(cache_dir, mini_dir)
    raise ValueError(f"No known session for user={user}; only 1 and 2 are defined.")
