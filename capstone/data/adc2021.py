"""Chit 2: unsupervised new-physics detection at the LHC Level-1 trigger (FPGA-L1), the ADC2021 dataset.

    from capstone.data.adc2021 import load, efficiency_at_fpr, SIGNALS
    X, _ = load("train")          # X: (2000000, 57) float32, background only. Training gets NO labels (y is None)
    X, y = load("val")            # y: 0 = background (Standard Model), 1..4 = the signal in SIGNALS[y - 1]

**Source.** Govorkova et al., *LHC physics dataset for unsupervised New Physics detection at 40 MHz*, Sci. Data 9, 118 (2022), arXiv:2107.02157.
Zenodo: background 5046389; signals (the "_filtered" versions, which match the background's selection) A→4ℓ 7152590, LQ→bτ 7152599, h⁰→ττ 7152614, h±→τν 7152617.
**Licence:** **CC BY 4.0** (all five Zenodo records). **Cite:** the paper above and the Zenodo records.

**What one event is.** What an L1 trigger sees: 19 physics objects, in fixed slots: missing transverse energy (slot 0), up to 4 electrons (1-4), up to 4 muons (5-8)
and up to 10 jets (9-18), each with (pT [GeV], η, φ). Empty slots are zeros. One input = **57 numbers**, ordered slot by slot [pT, η, φ, pT, η, φ, ...]
(MET's η is always 0). The original 4th column (object type) is dropped: it is implied by the slot. No other preprocessing: scale the inputs yourself
(e.g. pT on a log or power-of-two scale), using the **training** split only.

**Splits (fixed).** The 4,000,000 background events are permuted once (seed 0): `train` = 2,000,000, `val` = 1,000,000, `test` = 1,000,000.
Each signal sample is permuted once (seed 1) and split in half between `val` and `test`. **`train` is background only and has no labels:** the method must be
unsupervised. You may look at the signals in `val` to choose a design; report the final numbers on `test`.

**Metric.** Signal efficiency (true-positive rate) at a false-positive rate of **1e-5**, the standard working point for this dataset (the ADC2021 challenge):
of the 40 MHz collision rate, only ~400 Hz of background may pass this trigger (see `efficiency_at_fpr`). With 1,000,000 background events in `test`,
the threshold is set by the top 10 of them, so differences of a few tenths of a percent are noise: compare on the full `test` split only.

**Download.** The first call fetches `adc2021.npz` (~0.25 GB) from the course's GitHub release. The fallback rebuilds it from Zenodo (~0.3 GB of HDF5; Zenodo can be slow).
"""
import numpy as np
from capstone.data import fetch

CHIT = "adc2021"
NAME = "adc2021.npz"
ZENODO = "https://zenodo.org/records"
BACKGROUND = ("5046389", "background_for_training.h5")
SIGNALS = ["A→4ℓ", "LQ→bτ", "h⁰→ττ", "h±→τν"]
SIGNAL_FILES = [("7152590", "Ato4l_lepFilter_13TeV_filtered.h5"), ("7152599", "leptoquark_LOWMASS_lepFilter_13TeV_filtered.h5"),
                ("7152614", "hToTauTau_13TeV_PU20_filtered.h5"), ("7152617", "hChToTauNu_13TeV_PU20_filtered.h5")]
OBJECTS = ["MET"] + [f"e{i}" for i in range(4)] + [f"mu{i}" for i in range(4)] + [f"jet{i}" for i in range(10)]
FEATURES = [f"{o}_{v}" for o in OBJECTS for v in ("pt", "eta", "phi")]
N_TRAIN, N_VAL = 2_000_000, 1_000_000
FPR = 1e-5                    # the dataset's standard working point (~400 Hz of the 40 MHz)
KEYS = ["X_train", "X_val", "y_val", "X_test", "y_test"]


def _read(path):
    import h5py
    with h5py.File(path, "r") as h:
        P = h["Particles"][:]                                  # (n, 19, 4): pT, eta, phi, object type
    return P[:, :, :3].reshape(len(P), 57).astype(np.float32)


def build(path, src_dir=None):
    """Rebuild adc2021.npz from the five Zenodo HDF5 files (downloaded into `src_dir` if missing)."""
    from pathlib import Path
    src_dir = Path(src_dir or fetch.chit_dir(CHIT) / "zenodo")
    files = []
    for rec, f in [BACKGROUND] + SIGNAL_FILES:
        if not (src_dir / f).exists() and not fetch.download(f"{ZENODO}/{rec}/files/{f}?download=1", src_dir / f, f"{f} from Zenodo"):
            raise RuntimeError(f"could not download {f} from Zenodo record {rec}")
        files.append(src_dir / f)
    B = _read(files[0])
    B = B[np.random.default_rng(0).permutation(len(B))]
    parts = {"train": B[:N_TRAIN], "val": [B[N_TRAIN:N_TRAIN + N_VAL]], "test": [B[N_TRAIN + N_VAL:]]}
    labels = {"val": [np.zeros(N_VAL, np.int8)], "test": [np.zeros(len(B) - N_TRAIN - N_VAL, np.int8)]}
    for k, f in enumerate(files[1:], start=1):
        S = _read(f)
        S = S[np.random.default_rng(1).permutation(len(S))]
        h = len(S) // 2
        for s, part in (("val", S[:h]), ("test", S[h:])):
            parts[s].append(part); labels[s].append(np.full(len(part), k, np.int8))
    out = {"X_train": parts["train"]}
    for s in ("val", "test"):                                   # mix background and signals once, so any prefix (max_items) is representative
        X, y = np.concatenate(parts[s]), np.concatenate(labels[s])
        p = np.random.default_rng(2).permutation(len(y))
        out[f"X_{s}"], out[f"y_{s}"] = X[p], y[p]
    np.savez_compressed(path, **out)


def load(split="train", max_items=None):
    """Returns (X, y). X: (n, 57) float32. y: None for `train` (background only, unsupervised); for `val`/`test` an int64 array,
    0 = background, k = 1..4 = signal SIGNALS[k - 1]. `max_items` keeps a fixed prefix (val/test are pre-mixed, so a prefix is representative;
    but fewer background events make the 1e-5 threshold very noisy: report final numbers on the full test split)."""
    fetch.check(split)
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or 50_000
        y = None if split == "train" else rng.choice(5, n, p=[0.8, 0.05, 0.05, 0.05, 0.05])
        X = np.abs(rng.normal(0, 20, (n, 57))) * (1 if y is None else (1 + (y[:, None] > 0)))
        return X.astype(np.float32), (None if y is None else y.astype(np.int64))
    with np.load(fetch.get_npz(CHIT, NAME, KEYS, build)) as z:
        X = z[f"X_{split}"]
        y = None if split == "train" else z[f"y_{split}"].astype(np.int64)
    X, y = fetch.take(split, (X, y), max_items)
    return X, y


def efficiency_at_fpr(score, y, fpr=FPR):
    """Anomaly `score` (higher = more anomalous) on val/test with labels `y`. Sets the threshold so that a fraction `fpr` of the background passes,
    and returns {signal name: fraction of that signal passing}."""
    score, y = np.asarray(score), np.asarray(y)
    thr = np.quantile(score[y == 0], 1 - fpr)
    return {name: float(np.mean(score[y == k] > thr)) for k, name in enumerate(SIGNALS, start=1)}
