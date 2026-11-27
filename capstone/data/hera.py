"""Chit 3: flagging radio-frequency interference (RFI) in HERA data on a real-time GPU (GPU-RT). HERA, Karoo, South Africa.

    from capstone.data.hera import load, threshold_flagger, auprc, best_f1
    X, y = load("train")          # X: (360, 1, 512, 512) float32 visibility amplitudes, y: (360, 1, 512, 512) uint8 RFI mask (1 = RFI)
    s = threshold_flagger(X[:1])  # the classical baseline to beat: a robust score per pixel (higher = more likely RFI)

**Source.** Mesarcik et al., *Learning to detect radio frequency interference in radio astronomy without seeing it*, MNRAS 516, 5367 (2022),
arXiv:2207.00351. Data: Zenodo record 6724065 (https://zenodo.org/records/6724065), file `HERA_04-03-2022_all.pkl`: simulated HERA visibilities
(hera_sim) with all RFI types (stations, DTV, impulses, scatter) and **exact** per-pixel RFI masks. **Licence:** **CC BY 4.0** (Zenodo record metadata).
**Cite:** the paper above and the Zenodo record.

**What one sample is.** One "waterfall": the amplitude |V| of one baseline's visibility on a 512 × 512 time-frequency grid (axes as stored in the
record), and its mask. 2.75% of the pixels are RFI, in every split. The masks contain lines along both axes (narrow-band transmitters that are always on,
and broadband impulses that are short), so look at a few waterfalls before you design for one orientation.

**Splits (fixed).** The record ships 420 training and 140 test waterfalls. `test` is the published test set, unchanged. `val` is 60 of the 420 training
waterfalls (fixed random choice, seed 0); `train` is the other 360.

**Metric.** Per-pixel AUPRC (average precision, threshold-free; see `auprc`) and F1 at a threshold chosen on `val` (see `best_f1`), on the test set.

**Preprocessing.** None: you get the amplitudes as simulated (float16 in the file, float32 here). Their dynamic range is large (10^-5 to ~400), so most
methods work on a log scale. Whatever you do to the input runs inside your latency budget: put it in the timed function.

**Download.** The first call fetches `hera.npz` (~0.11 GB) from the course's GitHub release. The fallback rebuilds it from Zenodo (0.44 GB; Zenodo can be slow).
The Zenodo file is a Python pickle: it is read with a restricted unpickler that only accepts numpy arrays.
"""
import pickle
import numpy as np
from capstone.data import fetch

CHIT = "hera"
NAME = "hera.npz"
ZENODO_URL = "https://zenodo.org/records/6724065/files/HERA_04-03-2022_all.pkl?download=1"
PKL = "HERA_04-03-2022_all.pkl"
N_VAL = 60
KEYS = [f"{a}_{s}" for s in fetch.SPLITS for a in ("X", "y")]


class _NumpyOnly(pickle.Unpickler):
    """Only numpy array reconstruction is allowed: anything else in the file raises instead of running."""
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to load {module}.{name} from the HERA pickle")


def build(path):
    """Rebuild hera.npz from the Zenodo pickle."""
    src = fetch.chit_dir(CHIT) / PKL
    if not src.exists() and not fetch.download(ZENODO_URL, src, f"{PKL} from Zenodo (0.44 GB)"):
        raise RuntimeError("could not download the HERA data from Zenodo")
    with open(src, "rb") as f:
        Xtr, ytr, Xte, yte = _NumpyOnly(f).load()                    # (n, 512, 512, 1): float16 amplitudes, bool masks
    chw = lambda a: np.ascontiguousarray(np.moveaxis(a, -1, 1))       # -> (n, 1, 512, 512)
    perm = np.random.default_rng(0).permutation(len(Xtr))
    va, tr = np.sort(perm[:N_VAL]), np.sort(perm[N_VAL:])
    np.savez_compressed(path, X_train=chw(Xtr[tr]), y_train=chw(ytr[tr]).astype(np.uint8), X_val=chw(Xtr[va]), y_val=chw(ytr[va]).astype(np.uint8),
                        X_test=chw(Xte), y_test=chw(yte).astype(np.uint8))


def load(split="train", max_items=None):
    """Returns (X, y): X (n, 1, 512, 512) float32 amplitudes, y (n, 1, 512, 512) uint8 masks (1 = RFI)."""
    fetch.check(split)
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or {"train": 36, "val": 6, "test": 14}[split]
        y = np.zeros((n, 1, 512, 512), np.uint8)
        y[:, :, :, rng.integers(0, 512, 12)] = 1                         # a few "stations" (constant-frequency lines)
        X = (np.abs(rng.normal(1, 0.3, y.shape)) * (1 + 20 * y)).astype(np.float32)
        return X, y
    with np.load(fetch.get_npz(CHIT, NAME, KEYS, build)) as z:
        X, y = z[f"X_{split}"], z[f"y_{split}"]
    X, y = fetch.take(split, (X, y), max_items)
    return X.astype(np.float32), y


def threshold_flagger(X):
    """The classical baseline: a robust z-score per pixel. For every row of the waterfall, the median and the MAD along the last axis are computed
    on the log amplitude; score = (log|V| - median) / (1.4826 * MAD). Flag pixels whose score exceeds a threshold (choose it on val with `best_f1`).
    (Along the last axis it does clearly better than along the other one: AUPRC 0.45 vs. 0.26 on val.)
    X: numpy array or torch tensor, (n, 1, time, freq). Runs where X lives (CPU or GPU); returns the same type as X."""
    import torch
    t = torch.as_tensor(X).float()
    lx = torch.log10(t + 1e-6)
    med = lx.median(dim=3, keepdim=True).values
    mad = (lx - med).abs().median(dim=3, keepdim=True).values
    s = (lx - med) / (1.4826 * mad + 1e-6)
    return s if isinstance(X, torch.Tensor) else s.numpy()


def auprc(y, score, max_pixels=20_000_000):
    """Per-pixel average precision. Uses a fixed random subset of at most `max_pixels` pixels (for speed; the estimate is stable at that size)."""
    from sklearn.metrics import average_precision_score
    y, score = np.asarray(y).ravel(), np.asarray(score, np.float32).ravel()
    if len(y) > max_pixels:
        idx = np.random.default_rng(0).choice(len(y), max_pixels, replace=False)
        y, score = y[idx], score[idx]
    return float(average_precision_score(y, score))


def best_f1(y, score, n_thresholds=200):
    """(best F1, threshold) over `n_thresholds` quantiles of the score: choose the threshold on val, then apply it to test with `f1_at`."""
    y, score = np.asarray(y).ravel().astype(bool), np.asarray(score, np.float32).ravel()
    sub = np.random.default_rng(0).choice(len(score), min(len(score), 2_000_000), replace=False)
    best = (0.0, None)
    for thr in np.unique(np.quantile(score[sub], np.linspace(0.5, 0.9999, n_thresholds))):
        f = f1_at(y, score, thr)
        best = max(best, (f, float(thr)), key=lambda b: b[0])
    return best


def f1_at(y, score, thr):
    y, p = np.asarray(y).ravel().astype(bool), np.asarray(score).ravel() > thr
    tp = np.sum(y & p)
    return float(2 * tp / max(1, 2 * tp + np.sum(~y & p) + np.sum(y & ~p)))
