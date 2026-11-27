"""Chit 1: single-shot readout of a superconducting qubit from raw ADC traces (FPGA-ns).

    from capstone.data.qubit import load, fidelity
    X, y = load("train")          # X: (800000, 800) float32, y: (800000,) int64 in {0 = |g>, 1 = |e>}

**Source.** Di Guglielmo, Campos, et al., *End-to-end workflow for machine learning-based qubit readout with QICK and hls4ml* (2025).
Data: Zenodo record 14427490 (https://zenodo.org/records/14427490), measured on a QICK system (Xilinx ZCU216 RFSoC).
Code: https://github.com/fastmachinelearning/ml-quantum-readout (the `final/train.ipynb` notebook, whose preprocessing we copy).
**Licence:** the data is **CC BY 4.0** (Zenodo record metadata). The code repository has no licence file; we reuse only its preprocessing recipe, re-written here.
**Cite:** the paper above and the Zenodo record.

**What one sample is.** One readout of the qubit after preparing it in |g> (label 0) or |e> (label 1). The ADC records the in-phase (I) and quadrature (Q)
components of the returned microwave pulse; the Zenodo files hold 770 time samples per shot, stored interleaved as [I0, Q0, I1, Q1, ...] (1540 values).
Following the repository, we keep the readout window from sample 100 to sample 500 (400 time samples), so one input is **800 numbers** (I/Q interleaved).
A shorter window means a faster readout: that is one of your levers, so `load()` also lets you cut the window further (`window=`), but never widen it.

**Splits (fixed).** The Zenodo record ships a 900,000-shot training file and a 100,000-shot test file. `test` is that test file, unchanged.
`val` is 100,000 shots drawn from the training file with a fixed random permutation (seed 0); `train` is the remaining 800,000. Shots are independent
measurements, so a random split does not leak. The two classes are balanced.

**Metric.** Readout fidelity as defined in the repository: the mean of the per-state accuracies, F = (P(0|g) + P(1|e)) / 2 (see `fidelity`).

**Download.** The first call fetches `qubit.npz` (~1.4 GB) from the course's GitHub release. The fallback rebuilds it from Zenodo,
which means downloading 12.3 GB (slow: Zenodo often serves ~1 MB/s); the 11 GB training file is memory-mapped, not loaded, so the build needs ~2 GB of working memory (about 1 min).
"""
import numpy as np
from capstone.data import fetch

CHIT = "qubit"
NAME = "qubit.npz"
ZENODO = "https://zenodo.org/records/14427490/files"
FILES = {"X_train": "0528_X_train_0_770.npy", "y_train": "0528_y_train_0_770.npy",
         "X_test": "0528_X_test_0_770.npy", "y_test": "0528_y_test_0_770.npy"}
START, STOP = 100, 500            # readout window (time samples), as in ml-quantum-readout/final/train.ipynb
N_FEATURES = 2 * (STOP - START)   # I and Q per time sample
N_VAL = 100_000
KEYS = [f"{a}_{s}" for s in fetch.SPLITS for a in ("X", "y")]


def _window(X):
    return X[:, 2 * START: 2 * STOP]


def build(path, src_dir=None):
    """Rebuild qubit.npz from the four Zenodo .npy files (downloaded into `src_dir` if missing)."""
    from pathlib import Path
    src_dir = Path(src_dir or fetch.chit_dir(CHIT) / "zenodo")
    for f in FILES.values():
        if not (src_dir / f).exists() and not fetch.download(f"{ZENODO}/{f}?download=1", src_dir / f, f"{f} from Zenodo"):
            raise RuntimeError(f"could not download {f} from Zenodo")
    Xtr = np.load(src_dir / FILES["X_train"], mmap_mode="r")          # (900000, 1540) float64 on disk; never read in full
    ytr = np.load(src_dir / FILES["y_train"]).astype(np.int64)
    Xte = np.load(src_dir / FILES["X_test"], mmap_mode="r")
    yte = np.load(src_dir / FILES["y_test"]).astype(np.int64)
    perm = np.random.default_rng(0).permutation(len(ytr))
    parts = {"val": np.sort(perm[:N_VAL]), "train": np.sort(perm[N_VAL:])}

    def gather(X, idx, chunk=50_000):
        """The readout window of the rows `idx`, chunk by chunk, stored as int16: the ADC counts are integers well inside the int16 range."""
        out = []
        for i in range(0, len(idx), chunk):
            w = _window(X[idx[i:i + chunk]])
            assert np.all(w == np.round(w)) and np.abs(w).max() < 2**15, "unexpected non-integer ADC values"
            out.append(w.astype(np.int16))
        return np.concatenate(out)

    np.savez_compressed(path, X_train=gather(Xtr, parts["train"]), y_train=ytr[parts["train"]],
                        X_val=gather(Xtr, parts["val"]), y_val=ytr[parts["val"]],
                        X_test=gather(Xte, np.arange(len(yte))), y_test=yte)


def load(split="train", max_items=None, window=(START, STOP)):
    """Returns (X, y): X (n, 2 * (stop - start)) float32, I/Q interleaved; y (n,) int64, 0 = |g>, 1 = |e>.
    `window=(start, stop)` in time samples must lie inside (100, 500): a shorter window is a faster readout."""
    fetch.check(split)
    a, b = window
    if not START <= a < b <= STOP:
        raise ValueError(f"window must lie inside ({START}, {STOP})")
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or 20_000
        y = rng.integers(0, 2, n)
        X = rng.normal(0, 30, (n, N_FEATURES)) + 15 * y[:, None] * np.sin(np.arange(N_FEATURES) / 40)[None]
    else:
        with np.load(fetch.get_npz(CHIT, NAME, KEYS, build)) as z:
            X, y = z[f"X_{split}"], z[f"y_{split}"]
    X, y = fetch.take(split, (X, y), max_items)
    X = X[:, 2 * (a - START): 2 * (b - START)]
    return X.astype(np.float32), y.astype(np.int64)


def fidelity(y_true, y_pred):
    """Readout fidelity F = (P(0|g) + P(1|e)) / 2."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return 0.5 * (np.mean(y_pred[y_true == 0] == 0) + np.mean(y_pred[y_true == 1] == 1))
