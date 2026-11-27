"""Shared plumbing for the capstone data loaders: where data lives, and the fast-download-with-fallback pattern of `common/jets.py`.

Each chit's data is kept in `$FASTML_DATA/<chit>/` (default: the repo's `data/` folder, the same place as `jets.npz`).
The first `load()` downloads a compact, ready-made `.npz` from the course's GitHub release (fast). If that fails, the loader
rebuilds the same file from the original source (slower, sometimes much slower) and keeps it, so every later call works offline.

On Colab, set USE_DRIVE = True in the setup cell to keep the data on your Google Drive between sessions (FASTML_DATA then points there).
`FASTML_SYNTHETIC=1` makes every loader return random stand-in data of the right shapes: for testing code only, never for results.
"""
import os, time, urllib.request
from pathlib import Path
import numpy as np

DATA_DIR = Path(os.environ.get("FASTML_DATA", Path(__file__).resolve().parents[2] / "data"))
RELEASE_URL = os.environ.get("FASTML_CAPSTONE_URL", "https://github.com/SagarA17/aims-fastml-labs/releases/download/data-capstone-v1")
SPLITS = ("train", "val", "test")


def synthetic():
    return os.environ.get("FASTML_SYNTHETIC") == "1"


def chit_dir(chit):
    d = DATA_DIR / chit
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url, path, what=""):
    """Download `url` to `path` (via a .part file). Returns True on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    try:
        print(f"Downloading {what or path.name} ...", flush=True)
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "aims-fastml-course"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        tmp.replace(path)
        print(f"  done: {path.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s -> {path}")
        return True
    except Exception as e:
        print(f"  failed ({e})")
        return False
    finally:
        tmp.unlink(missing_ok=True)


def valid_npz(path, keys):
    """True if `path` is an npz holding all `keys` (no pickles are ever loaded)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return all(k in z.files for k in keys)
    except Exception:
        return False


def get_npz(chit, name, keys, build):
    """Return the path of `$FASTML_DATA/<chit>/<name>`: already there, else from the GitHub release, else `build(path)` from the original source."""
    path = chit_dir(chit) / name
    if valid_npz(path, keys):
        return path
    if download(f"{RELEASE_URL}/{name}", path, f"{name} from the course's GitHub release") and valid_npz(path, keys):
        return path
    path.unlink(missing_ok=True)
    print(f"Building {name} from the original source instead (one time; see the loader's docstring for how long this takes).")
    build(path)
    assert valid_npz(path, keys), f"{path} was not built correctly"
    return path


def check(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")


def take(split, arrays, max_items=None):
    """Check the split name and cut every array to `max_items` (a fixed prefix, so repeated calls agree)."""
    check(split)
    return tuple(None if a is None else (a if max_items is None else a[:max_items]) for a in arrays)
