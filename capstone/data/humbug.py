"""Chit 5: acoustic mosquito detection for malaria surveillance on a microcontroller (MCU), from HumBugDB.

    from capstone.data.humbug import load, logmel, logmel_macs, recall_at_false_alarm, pr_auc
    A, y = load("train")                      # A: (n, 15360) int16, 1.92 s of audio at 8 kHz; y: (n,) int64, 1 = mosquito, 0 = no mosquito
    X = logmel(A)                             # (n, 1, 128, 30) float32: the paper's log-mel features (or compute your own)
    p = profile(model, (1, 128, 30), extra_macs=logmel_macs())   # the feature extraction counts toward the MCU budget

**Source.** Kiskin et al., *HumBugDB: a large-scale acoustic mosquito dataset*, NeurIPS 2021 Datasets and Benchmarks track, arXiv:2110.07607.
Audio and metadata: Zenodo record 4904800 (https://zenodo.org/records/4904800, version 0.0.1, the one the paper uses). Code, splits and features:
https://github.com/HumBug-Mosquito/HumBugDB (`lib/config.py`, `lib/feat_util.py`, `notebooks/main.ipynb`).
**Licence:** the data is **CC BY 4.0** (Zenodo record metadata); the code repository is MIT (we re-implement its feature recipe, we don't copy it).
**Cite:** the paper above and the Zenodo record.

**What one sample is.** A window of **1.92 s of audio at 8 kHz** (15,360 int16 samples): the window and sample rate of the paper's pipeline
(`rate = 8000`, 30 spectrogram frames × 512-sample hop). Label 1 = the window lies in a clip labelled *mosquito*, 0 = background or other audio (speech,
other insects, rural and urban noise; the paper's `audio` and `background` types). Clips recorded at 44.1 kHz are resampled to 8 kHz (polyphase filter);
clips shorter than 1.92 s are dropped, as in the paper, and so are the clips whose audio length disagrees with the metadata by more than 1% (the paper skips them too).
Within a clip, windows start at 0, 1.92 s, 3.84 s, ... (`step=` gives overlapping training windows: more data, more memory).

**Splits (fixed, never one recording in two splits).** `test` = the paper's two test sets, unchanged: **A** = field recordings from Tanzania (Ifakara),
**B** = lab cultures in the UK (Oxford Zoology and LSTMH, London); `meta("test")["test_set"]` tells them apart. These are whole experiments,
so nothing from their sites and setups is in training. The paper trains on everything else (it takes validation windows at random from that, which leaks
between overlapping windows of one recording); we split it **by recording** (the metadata's `name`) instead: within each of the five training experiments,
the recordings are shuffled once (seed 0) and taken into `val` until it holds 15% of that experiment's seconds (a recording longer than 30% of its
experiment, e.g. the single 19-min background recording of the CDC cultures, always stays in `train`); the rest is `train`.
So `val` shares sites with `train` (it is optimistic) and `test` doesn't: expect test numbers below val numbers.

**Metric.** Mosquito recall (true-positive rate) when at most **1% of the background windows** raise a false alarm (`recall_at_false_alarm`), and PR-AUC
(`pr_auc`). Why the false-alarm rate: in the field only ~1-2% of the night contains a mosquito, so at the test set's ~50% mosquito share every false alarm
would weigh ~50 × more in deployment. `labels="species"` gives the species instead (`SPECIES`; stretch goal, e.g. *Anopheles* vs. other: `is_anopheles`).

**Features.** `logmel` is the paper's "Feat. B" (librosa defaults at 8 kHz): 2048-point FFT with a periodic Hann window, hop 512, zero-padded "centred" frames,
power spectrum, 128 Slaney mel bands (0-4 kHz), dB with an 80 dB floor, then standardised. The one change: the paper standardises each *clip*; a device sees
one window at a time, so we standardise each *window*. `logmel_macs()` counts its multiply-accumulates with the same parameters (like `mfcc_macs` elsewhere):
the FFT size, hop and number of bands are levers, and so is the sample rate (mosquito wingbeats are ~300-1,000 Hz plus harmonics).

**Download.** The first call fetches `humbug_train.npz` (~0.9 GB) and `humbug_eval.npz` (~0.26 GB) from the course's GitHub release. The fallback rebuilds them
from Zenodo: 4.1 GB of zipped WAV files (Zenodo often serves ~1 MB/s: about an hour), then ~2 min of decoding and resampling.
In memory, `load("train")` is ~1.4 GB of int16 audio (45k windows); its log-mel features are ~0.7 GB of float32.
"""
import io
import numpy as np
from capstone.data import fetch

CHIT = "humbug"
ZENODO = "https://zenodo.org/records/4904800/files"
CSV = "neurips_2021_zenodo_0_0_1.csv"
ZIPS = [f"humbugdb_neurips_2021_{i}.zip" for i in range(1, 5)]
SR = 8000
WINDOW = 15_360                               # 1.92 s = 30 frames x 512 hop at 8 kHz (HumBugDB lib/config.py)
VAL_SHARE = 0.15
FILES = {"train": "humbug_train.npz", "val": "humbug_eval.npz", "test": "humbug_eval.npz"}
SPECIES = None                                # filled from the file on first load: index 0 = "none" (no mosquito), then the species names
_SPLIT_KEYS = ("audio", "start", "length", "label", "species", "experiment", "clip_id")


def _keys(split):
    return [f"{k}_{split}" for k in _SPLIT_KEYS] + ["species_names", "experiment_names"]


def _test_set(row):
    if row["country"] == "Tanzania" and row["location_type"] == "field":
        return "A"
    if row["country"] == "UK" and row["location_type"] == "culture":
        return "B"
    return ""


def plan(rows):
    """rows: the metadata CSV as dicts. Returns {split: [rows]} (clips >= 1.92 s only), splitting the non-test experiments by recording."""
    rows = [r for r in rows if float(r["length"]) > WINDOW / SR]
    out = {"train": [], "val": [], "test": [r for r in rows if _test_set(r)]}
    rest = [r for r in rows if not _test_set(r)]
    rng = np.random.default_rng(0)
    for exp in sorted({_experiment(r) for r in rest}):
        mine = [r for r in rest if _experiment(r) == exp]
        sec = {}
        for r in mine:
            sec[r["name"]] = sec.get(r["name"], 0.0) + float(r["length"])
        names = sorted(sec)
        names = [names[i] for i in rng.permutation(len(names))]
        val, acc, total = set(), 0.0, sum(sec.values())
        for n in names:
            if acc >= VAL_SHARE * total:
                break
            if sec[n] > 2 * VAL_SHARE * total:             # one long recording would dominate val: it stays in train
                continue
            val.add(n); acc += sec[n]
        for r in mine:
            out["val" if r["name"] in val else "train"].append(r)
    return out


def _experiment(r):
    return f"{r['country']} / {r['location_type']} / {r['place']}"


def _decode(zip_path, wanted):
    """{clip id: int16 audio at 8 kHz} for the clips of `wanted` (id -> metadata length) found in one zip; clips whose length disagrees by > 1% are skipped."""
    import zipfile
    import soundfile as sf
    from math import gcd
    from scipy.signal import resample_poly
    got, bad = {}, []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if stem not in wanted:
                continue
            x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
            if abs(len(x) / sr - wanted[stem]) > 0.01 * wanted[stem]:
                bad.append(stem); continue
            x = x.mean(1)
            if sr != SR:
                g = gcd(SR, sr)
                x = resample_poly(x, SR // g, sr // g)
            got[stem] = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    return got, bad


def build(path_train, path_eval=None, src_dir=None):
    """Rebuild humbug_train.npz and humbug_eval.npz from the Zenodo files (downloaded into `src_dir` if missing)."""
    import csv
    from pathlib import Path
    from concurrent.futures import ProcessPoolExecutor
    path_train = Path(path_train)
    path_eval = Path(path_eval or path_train.with_name(FILES["test"]))
    src_dir = Path(src_dir or fetch.chit_dir(CHIT) / "zenodo")
    for f in [CSV] + ZIPS:
        if not (src_dir / f).exists() and not fetch.download(f"{ZENODO}/{f}?download=1", src_dir / f, f"{f} from Zenodo"):
            raise RuntimeError(f"could not download {f} from Zenodo")
    with open(src_dir / CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    p = plan(rows)
    wanted = {r["id"]: float(r["length"]) for split in p.values() for r in split}
    audio, bad = {}, []
    with ProcessPoolExecutor(len(ZIPS)) as ex:
        for got, b in ex.map(_decode, [src_dir / z for z in ZIPS], [wanted] * len(ZIPS)):
            audio.update(got); bad += b
    print(f"  {len(audio)} clips decoded, {len(bad)} skipped (length disagrees with the metadata), {len(wanted) - len(audio) - len(bad)} not found")
    species = ["none"] + sorted({r["species"] for r in rows if r["sound_type"] == "mosquito" and r["species"]}) + ["unknown"]
    experiments = sorted({_experiment(r) for r in rows})
    out = {"train": {}, "eval": {}}
    for split, items in p.items():
        items = [r for r in items if r["id"] in audio]
        lens = np.array([len(audio[r["id"]]) for r in items], np.int64)
        d = out["train" if split == "train" else "eval"]
        d[f"audio_{split}"] = np.concatenate([audio[r["id"]] for r in items])
        d[f"start_{split}"] = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        d[f"length_{split}"] = lens
        d[f"label_{split}"] = np.array([r["sound_type"] == "mosquito" for r in items], np.int8)
        d[f"species_{split}"] = np.array([0 if r["sound_type"] != "mosquito" else species.index(r["species"] or "unknown") for r in items], np.int16)
        d[f"experiment_{split}"] = np.array([experiments.index(_experiment(r)) for r in items], np.int8)
        d[f"clip_id_{split}"] = np.array([int(r["id"]) for r in items], np.int64)
        print(f"  {split}: {len(items)} clips, {lens.sum() / SR / 3600:.2f} h, {len({r['name'] for r in items})} recordings")
    for path, d in ((path_train, out["train"]), (path_eval, out["eval"])):
        np.savez_compressed(path, species_names=np.array(species), experiment_names=np.array(experiments), **d)


def _file(split):
    def build_both(path):
        build(fetch.chit_dir(CHIT) / FILES["train"], fetch.chit_dir(CHIT) / FILES["test"])
    return fetch.get_npz(CHIT, FILES[split], _keys(split), build_both)


def _windows(start, length, step):
    """(window start sample, clip index) for every full window of every clip."""
    s, c = [], []
    for k, (a, n) in enumerate(zip(start, length)):
        offs = np.arange(0, n - WINDOW + 1, step)
        s.append(a + offs); c.append(np.full(len(offs), k))
    return np.concatenate(s), np.concatenate(c)


def _index(split, step):
    global SPECIES
    with np.load(_file(split)) as z:
        clips = {k: z[f"{k}_{split}"] for k in _SPLIT_KEYS if k != "audio"}
        SPECIES = list(z["species_names"])
        experiments = list(z["experiment_names"])
    starts, clip = _windows(clips["start"], clips["length"], step)
    perm = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split]).permutation(len(starts))   # mixed once, so any prefix is representative
    return starts[perm], clip[perm], clips, experiments


def load(split="train", max_items=None, labels="mosquito", step=None):
    """Returns (A, y). A: (n, 15360) int16 audio windows (1.92 s at 8 kHz; divide by 32768 for [-1, 1]). y: (n,) int64,
    labels="mosquito": 1 = mosquito, 0 = no mosquito; labels="species": index into SPECIES (0 = no mosquito).
    step: window step in samples for `train` (default 15360, i.e. no overlap; e.g. 2560 = the paper's 0.32 s gives 6 x the windows and memory).
    val and test always use non-overlapping windows (the evaluation unit). Windows are in a fixed random order, so a `max_items` prefix is representative."""
    fetch.check(split)
    if labels not in ("mosquito", "species"):
        raise ValueError('labels must be "mosquito" or "species"')
    if step is not None and split != "train":
        raise ValueError("step= is for the training split only: val and test use non-overlapping 1.92-s windows")
    if fetch.synthetic():
        global SPECIES
        SPECIES = SPECIES or ["none", "an arabiensis", "culex quinquefasciatus", "unknown"]
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or 2_000
        y = rng.integers(0, 2, n)
        t = np.arange(WINDOW) / SR
        f0 = rng.uniform(350, 650, (n, 1))
        A = 0.05 * rng.normal(size=(n, WINDOW)) + 0.2 * y[:, None] * np.sin(2 * np.pi * f0 * t) * rng.uniform(0.2, 1, (n, 1))
        A = (np.clip(A, -1, 1) * 32767).astype(np.int16)
        return A, (y if labels == "mosquito" else y * rng.integers(1, len(SPECIES), n)).astype(np.int64)
    starts, clip, clips, _ = _index(split, step or WINDOW)
    starts, clip = fetch.take(split, (starts, clip), max_items)
    with np.load(_file(split)) as z:
        audio = z[f"audio_{split}"]
    A = np.empty((len(starts), WINDOW), np.int16)
    for i in range(0, len(starts), 1024):                  # in chunks, to keep the index array small
        A[i:i + 1024] = audio[starts[i:i + 1024, None] + np.arange(WINDOW)]
    y = clips["label" if labels == "mosquito" else "species"][clip]
    return A, y.astype(np.int64)


def meta(split, max_items=None, step=None):
    """Per-window metadata in the same order as load(split, max_items, step=step): dict of arrays
    clip_id (HumBugDB id), experiment (country / location type / place), test_set ("A", "B" or "" outside test), species (name or "none"),
    label (1 = mosquito, 0 = no mosquito, as in load(): handy for counting without loading the audio)."""
    fetch.check(split)
    if fetch.synthetic():
        n = max_items or 2_000
        test_set = np.where(np.arange(n) % 2, "B", "A") if split == "test" else np.array([""] * n)
        label = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split]).integers(0, 2, n)      # the same draw as load()
        return dict(clip_id=np.arange(n), experiment=np.array(["synthetic"] * n), test_set=test_set, species=np.array(["none"] * n), label=label)
    starts, clip, clips, experiments = _index(split, step or WINDOW)
    _, clip = fetch.take(split, (starts, clip), max_items)
    exp = np.array(experiments)[clips["experiment"][clip]]
    test_set = np.array(["A" if e.startswith("Tanzania / field") else "B" if e.startswith("UK / culture") else "" for e in exp])
    return dict(clip_id=clips["clip_id"][clip], experiment=exp, test_set=test_set, species=np.array(SPECIES)[clips["species"][clip]],
                label=clips["label"][clip].astype(np.int64))


def is_anopheles(species_index):
    """True for the Anopheles species (the malaria vectors) in SPECIES. Call load(..., labels="species") first."""
    return np.array([SPECIES[int(k)].startswith("an ") for k in np.atleast_1d(species_index)])


# ---------- features: the paper's log-mel ("Feat. B"), and its cost ----------
def _hz_to_mel(f):                          # Slaney's mel scale (librosa's default, htk=False)
    f = np.asarray(f, np.float64)
    return np.where(f < 1000, 3 * f / 200, 15 + np.log(np.maximum(f, 1e-9) / 1000) / (np.log(6.4) / 27))


def _mel_to_hz(m):
    m = np.asarray(m, np.float64)
    return np.where(m < 15, 200 * m / 3, 1000 * np.exp((np.log(6.4) / 27) * (m - 15)))


def mel_filterbank(n_fft=2048, n_mels=128, fmin=0.0, fmax=SR / 2, sr=SR):
    """(n_mels, n_fft // 2 + 1) triangular filters with Slaney normalisation (as librosa.filters.mel)."""
    mel_f = _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))
    fft_f = np.fft.rfftfreq(n_fft, 1 / sr)
    fdiff, ramps = np.diff(mel_f), mel_f[:, None] - fft_f[None]
    W = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1, None], ramps[2:] / fdiff[1:, None]))
    return (W * (2 / (mel_f[2:] - mel_f[:-2]))[:, None]).astype(np.float32)


def logmel(audio, n_fft=2048, hop=512, n_mels=128, frames=30, top_db=80.0):
    """audio: (n, samples) int16 or float at 8 kHz -> (n, 1, n_mels, frames) float32 standardised log-mel spectrograms.
    Frame t is centred on sample t * hop (zero padding at the edges); power |FFT|^2 with a periodic Hann window; dB relative to the window's maximum,
    floored at -top_db; then each window is standardised (zero mean, unit variance)."""
    audio = np.asarray(audio)
    x = audio.astype(np.float32) / 32768 if audio.dtype == np.int16 else audio.astype(np.float32)
    x = np.pad(x, ((0, 0), (n_fft // 2, n_fft // 2)))
    idx = np.arange(n_fft)[None] + hop * np.arange(frames)[:, None]
    w = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)).astype(np.float32)
    fb = mel_filterbank(n_fft, n_mels)
    out = np.empty((len(x), n_mels, frames), np.float32)
    for i in range(0, len(x), 512):                        # in chunks, to keep memory small
        spec = np.abs(np.fft.rfft(x[i:i + 512][:, idx] * w, n=n_fft)) ** 2      # (b, frames, bins)
        db = 10 * np.log10(np.maximum(spec @ fb.T, 1e-10))
        db = np.maximum(db, db.max((1, 2), keepdims=True) - top_db)
        db = (db - db.mean((1, 2), keepdims=True)) / (db.std((1, 2), keepdims=True) + 1e-6)
        out[i:i + 512] = db.transpose(0, 2, 1)
    return out[:, None]


def logmel_macs(n_fft=2048, hop=512, n_mels=128, frames=30):
    """Multiply-accumulates of `logmel` for one window, counted per frame as: window (n_fft) + real FFT as an n_fft/2-point complex radix-2 FFT plus
    its split step (4 real MACs per complex multiply: 4 * (n_fft/4 * log2(n_fft/2) + n_fft/2)) + power spectrum (2 per bin) + mel filterbank
    (its non-zero weights only); plus 3 per output value for the standardisation (mean, variance, scale). The log is not counted
    (like activations elsewhere). `hop` doesn't change the count for a fixed number of frames, but it sets how much audio the frames cover."""
    fft = 4 * (n_fft // 4 * int(np.log2(n_fft // 2)) + n_fft // 2)
    mel = int(np.count_nonzero(mel_filterbank(n_fft, n_mels)))
    return int(frames * (n_fft + fft + 2 * (n_fft // 2 + 1) + mel) + 3 * n_mels * frames)


# ---------- metric ----------
def recall_at_false_alarm(score, y, false_alarm=0.01):
    """`score`: higher = more likely a mosquito; y: 1 = mosquito. Sets the threshold so that a fraction `false_alarm` of the no-mosquito windows
    pass, and returns the fraction of mosquito windows that pass (recall)."""
    score, y = np.asarray(score), np.asarray(y)
    thr = np.quantile(score[y == 0], 1 - false_alarm)
    return float(np.mean(score[y == 1] > thr))


def pr_auc(score, y):
    """Area under the precision-recall curve (average precision), mosquito = positive class."""
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(np.asarray(y) == 1, score))
