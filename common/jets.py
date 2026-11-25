"""Shared data loader: the hls4ml LHC jet-tagging dataset (OpenML 'hls4ml_lhc_jets_hlf').

~830k jets, 16 high-level features, 5 classes (g, q, W, Z, t).

The first call downloads a ready-made `jets.npz` (~50 MB) from the course repo's GitHub release (fast), and falls back to
sklearn's `fetch_openml` (slow; the same call the official hls4ml tutorial makes) if that fails. The file is kept in
labs/data/ (or $FASTML_DATA), so every later call works offline.

Offline / classroom use: point FASTML_DATA at a folder holding `jets.npz` (e.g. a USB stick), and
SCIKIT_LEARN_DATA at a copy of a populated scikit_learn_data folder for the tutorial.
"""
import os
from pathlib import Path
import numpy as np

CLASSES = ["g", "q", "w", "z", "t"]
DATA_DIR = Path(os.environ.get("FASTML_DATA", Path(__file__).resolve().parent.parent / "data"))
DATA_URL = os.environ.get("FASTML_DATA_URL", "https://github.com/SagarA17/aims-fastml-labs/releases/download/data-v1/jets.npz")


def _valid(path: Path):
    """True if `path` holds the full dataset (no pickles are ever loaded)."""
    try:
        z = np.load(path, allow_pickle=False)
        return z["X"].shape == (830000, 16) and set(np.unique(z["y"])) == set(range(5)) and len(z["features"]) == 16
    except Exception:
        return False


def _download(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        import urllib.request, time
        print("Downloading the jet dataset from the course's GitHub release (~50 MB, one time)...")
        t0 = time.time()
        urllib.request.urlretrieve(DATA_URL, tmp)
        if _valid(tmp):
            tmp.replace(path)
            print(f"done: {path.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f} s, saved to {path}")
            return
        print("The downloaded file doesn't look right; falling back to OpenML.")
    except Exception as e:
        print(f"Fast download failed ({e}); falling back to OpenML.")
    finally:
        tmp.unlink(missing_ok=True)
    _download_openml(path)


def _download_openml(path: Path):
    from sklearn.datasets import fetch_openml
    print("Downloading the jet dataset from OpenML (one time; can take a few minutes)...")
    d = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=False, parser="auto")
    X = d.data.astype(np.float32)
    y = np.array([CLASSES.index(str(t)) for t in d.target], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, X=X, y=y, features=np.array(d.feature_names))


def _synthetic(n=200_000, seed=0):
    """Same shapes, random content. ONLY for testing the notebooks without network access."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    centers = rng.normal(0, 1.0, (5, 16)).astype(np.float32)
    X = (centers[y] + rng.normal(0, 1.5, (n, 16))).astype(np.float32)
    return X, y.astype(np.int64), np.array([f"f{i}" for i in range(16)])


def check_jets():
    """Pre-course check: download if needed, then verify. Returns True if the real dataset is ready."""
    path = DATA_DIR / "jets.npz"
    if not path.exists():
        _download(path)
    z = np.load(path)
    ok = z["X"].shape == (830000, 16) and set(np.unique(z["y"])) == set(range(5))
    print(f"jets.npz at {path}: X{z['X'].shape}, {len(np.unique(z['y']))} classes -> {'OK' if ok else 'UNEXPECTED'}")
    return ok


def load_jets(test_size=0.2, seed=42, standardize=True):
    """Returns (X_train, X_test, y_train, y_test, feature_names) as float32 / int64 numpy arrays."""
    if os.environ.get("FASTML_SYNTHETIC") == "1":
        X, y, feats = _synthetic()
    else:
        path = DATA_DIR / "jets.npz"
        if not path.exists():
            _download(path)
        z = np.load(path)
        X, y, feats = z["X"], z["y"], z["features"]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    Xtr, Xte = X[tr], X[te]
    if standardize:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xtr.astype(np.float32), Xte.astype(np.float32), y[tr], y[te], list(feats)
