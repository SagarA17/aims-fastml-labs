"""Chit 6: heartbeat classification for a wearable (MCU), MIT-BIH Arrhythmia Database.

    from capstone.data.mitbih import load, CLASSES, macro_f1
    X, y = load("train")          # X: (n, 1, 256) float32 in mV (lead MLII), y: (n,) int64 in 0..4 = CLASSES (N, S, V, F, Q)

**Source.** MIT-BIH Arrhythmia Database v1.0.0 on PhysioNet, https://physionet.org/content/mitdb/1.0.0/ : 48 half-hour two-lead ECG recordings
from 47 patients, 360 samples/s, every beat annotated by two cardiologists. **Licence:** **Open Data Commons Attribution License v1.0 (ODC-By 1.0)**.
**Cite (attribution required):** Moody GB, Mark RG, *The impact of the MIT-BIH Arrhythmia Database*, IEEE Eng. Med. Biol. 20(3):45-50 (2001);
and Goldberger AL et al., *PhysioBank, PhysioToolkit, and PhysioNet*, Circulation 101(23):e215-e220 (2000).

**What one sample is.** One beat: 256 samples (0.71 s) of lead MLII centred on the annotated R peak (128 before, 128 after), in millivolts, unfiltered.
Records 102 and 104 have no MLII lead (their pacemaker patients were recorded on V5/V2); for them the first lead is used. Beats too close to the start or
end of a recording for a full window, and non-beat annotations (rhythm changes, noise, ...), are skipped.

**Classes (AAMI EC57 grouping of the beat labels).** N normal & bundle-branch-block (N, L, R, e, j) · S supraventricular ectopic (A, a, J, S) ·
V ventricular ectopic (V, E) · F fusion (F) · Q paced & unclassifiable (/, f, Q). The classes are very unbalanced (N is ~90%): that's why the metric is macro-F1.

**Splits (fixed, patient-wise).** A wearable meets patients it was never trained on, so no patient is in two splits. We start from the standard
inter-patient split of de Chazal et al. (IEEE TBME 2004): DS1 for training, DS2 for testing. Two changes: record 202 moves to DS1, because 201 and 202
are the **same patient**; and the four paced records (which de Chazal excluded) are kept, so that class Q exists: 102 and 107 in train, 104 in val, 217 in test.
`val` is four DS1 records (106, 209, 223 and the paced 104), chosen so that val contains every class. See `RECORDS`.

**Metric.** Macro-F1 over the 5 classes on the test patients (see `macro_f1`).

**Download.** The first call fetches `mitbih.npz` (~30 MB) from the course's GitHub release; the fallback rebuilds it from PhysioNet (77 MB zip, needs `wfdb`; PhysioNet can be very slow: we saw 10-80 KB/s, i.e. up to 2 hours).
"""
import numpy as np
from capstone.data import fetch

CHIT = "mitbih"
NAME = "mitbih.npz"
ZIP_URL = "https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip"
CLASSES = ["N", "S", "V", "F", "Q"]
AAMI = {**dict.fromkeys("NLRej", 0), **dict.fromkeys("AaJS", 1), **dict.fromkeys("VE", 2), "F": 3, **dict.fromkeys("/fQ", 4)}
HALF = 128                                   # window = [R - 128, R + 128): 256 samples
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
VAL = [104, 106, 209, 223]
RECORDS = {"train": sorted(set(DS1 + [202, 102, 107]) - set(VAL)), "val": VAL, "test": sorted(set(DS2 + [217]) - {202})}
KEYS = [f"{a}_{s}" for s in fetch.SPLITS for a in ("X", "y")]


def beats(record_path):
    """All AAMI-labelled beat windows of one record: (X (n, 256) float32 mV, y (n,) int8)."""
    import wfdb
    rec, ann = wfdb.rdrecord(record_path), wfdb.rdann(record_path, "atr")
    ch = rec.sig_name.index("MLII") if "MLII" in rec.sig_name else 0
    sig = rec.p_signal[:, ch].astype(np.float32)
    X, y = [], []
    for r, s in zip(ann.sample, ann.symbol):
        if s in AAMI and HALF <= r <= len(sig) - HALF:
            X.append(sig[r - HALF: r + HALF]); y.append(AAMI[s])
    return np.array(X, np.float32), np.array(y, np.int8)


def build(path):
    """Rebuild mitbih.npz from the PhysioNet zip."""
    import zipfile
    src = fetch.chit_dir(CHIT) / "mitdb.zip"
    if not src.exists() and not fetch.download(ZIP_URL, src, "MIT-BIH from PhysioNet"):
        raise RuntimeError("could not download MIT-BIH from PhysioNet")
    out_dir = fetch.chit_dir(CHIT) / "mitdb"
    out_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(src) as z:
        for m in z.namelist():
            name = m.rsplit("/", 1)[-1]
            if name.split(".")[0].isdigit() and name.endswith((".dat", ".hea", ".atr")):
                (out_dir / name).write_bytes(z.read(m))
    out = {}
    for split, recs in RECORDS.items():
        parts = [beats(str(out_dir / str(r))) for r in recs]
        out[f"X_{split}"] = np.concatenate([p[0] for p in parts])[:, None]
        out[f"y_{split}"] = np.concatenate([p[1] for p in parts])
        out[f"record_{split}"] = np.concatenate([np.full(len(p[1]), r, np.int16) for r, p in zip(recs, parts)])
    np.savez_compressed(path, **out)


def load(split="train", max_items=None):
    """Returns (X, y): X (n, 1, 256) float32 in mV, y (n,) int64 in 0..4 (CLASSES). Beats are in recording order, record by record;
    `max_items` therefore keeps the first records only: shuffle yourself if you subsample for training."""
    fetch.check(split)
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or 10_000
        y = rng.choice(5, n, p=[0.85, 0.04, 0.07, 0.02, 0.02])
        t = np.linspace(-1, 1, 2 * HALF)
        X = (np.exp(-(t / (0.03 + 0.02 * y[:, None])) ** 2) + 0.05 * rng.normal(size=(n, 2 * HALF)))[:, None]
        return X.astype(np.float32), y.astype(np.int64)
    with np.load(fetch.get_npz(CHIT, NAME, KEYS, build)) as z:
        X, y = z[f"X_{split}"], z[f"y_{split}"].astype(np.int64)
    return fetch.take(split, (X, y), max_items)


def macro_f1(y_true, y_pred):
    """Unweighted mean of the per-class F1 scores over the 5 AAMI classes (a class absent from both y_true and y_pred counts as 0)."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(5)), average="macro", zero_division=0))
