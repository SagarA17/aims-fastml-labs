"""Chit 4: a camera-trap "wake-up" filter, animal vs. empty, on a microcontroller (MCU). Snapshot Karoo, Karoo National Park, South Africa.

    from capstone.data.karoo import load, recall_at_empties_kept
    X, y = load("train")                          # X: (n, 3, 96, 96) uint8 RGB, y: (n,) int64, 1 = animal, 0 = empty
    X, y = load("train", grayscale=True, size=64) # (n, 1, 64, 64): colour and resolution are your levers

**Source.** Snapshot Karoo, Season 1, part of Snapshot Safari, distributed by LILA BC: https://lila.science/datasets/snapshot-karoo
(metadata `SnapshotKaroo_S1_v1.0.json`, images in the public bucket gs://public-datasets-lila/snapshot-safari/KAR/KAR_public).
**Licence:** **Community Data License Agreement, permissive variant (CDLA-Permissive 1.0)**. **Cite:** Snapshot Karoo / Snapshot Safari (Pardo et al.,
*Snapshot Safari: a large-scale collaborative to monitor Africa's remarkable biodiversity*, S. Afr. J. Sci. 117 (2021)) and LILA BC.

**The full dataset** has 38,293 images in 14,889 sequences (bursts of 1-3 shots taken when the camera triggers), all 2592 × 2000 pixels.
By image: 31,792 empty (83%), 6,282 with animals (36 categories, incl. birds and reptiles), 219 "human" (not public, excluded here).
Labels are only reliable per **sequence**, so a sequence's label is given to all its images: 1 (animal) if any of its labels is an animal, 0 if it is empty.

**This file is a subset** (to keep the download small): **every** animal sequence (2,500 sequences, 6,282 images) plus a random sample of empty sequences
(seed 0) up to ~12,000 empty images. So ~65% of the images here are empty, vs. 83% in the wild. The metric below doesn't depend on that mix
(it is measured on the empties and the animals separately), but a model that outputs class probabilities is calibrated to this mix, not the field's.

**Preprocessing (done once, in `build`).** Each JPEG is decoded, centre-cropped to a square (2000 × 2000: 296 px cut from each side, the camera's info strip at the
bottom stays in), and downscaled to **96 × 96 RGB uint8** with area averaging. `load(size=..., grayscale=...)` reduces it further at load time.

**Splits (fixed).** By **sequence** (never by image: shots in one burst are near-duplicates): sequences are permuted once (seed 0) and split 70 / 15 / 15
into train / val / test, separately for animal and empty sequences (so every split has both). Cameras (sites) do appear in more than one split.

**Metric.** Animal recall when the filter keeps at most **5%** of the empty images (see `recall_at_empties_kept`; its `kept=` argument gives other operating points).

**Download.** The first call fetches `karoo.npz` (~0.4 GB) from the course's GitHub release. The fallback rebuilds it from LILA's bucket:
~16 GB of JPEGs, streamed (never kept), about 30-60 min.
"""
import io, json, zipfile
import numpy as np
from capstone.data import fetch

CHIT = "karoo"
NAME = "karoo.npz"
META_URL = "https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KAR/SnapshotKaroo_S1_v1.0.json.zip"
IMAGE_URL = "https://storage.googleapis.com/public-datasets-lila/snapshot-safari/KAR/KAR_public/"
SIZE = 96
N_EMPTY_IMAGES = 12_000
FRACTIONS = (0.70, 0.15, 0.15)
KEYS = [f"{a}_{s}" for s in fetch.SPLITS for a in ("X", "y")]


def _sequences(meta):
    """{seq_id: (label, [file names])} for every public sequence. label 1 = animal, 0 = empty; sequences with a human label are dropped."""
    cats = {c["id"]: c["name"] for c in meta["categories"]}
    img = {i["id"]: i for i in meta["images"]}
    names, files = {}, {}
    for i in meta["images"]:
        files.setdefault(i["seq_id"], []).append(i["file_name"])
    for a in meta["annotations"]:
        names.setdefault(img[a["image_id"]]["seq_id"], set()).add(cats[a["category_id"]])
    return {s: (int(n != {"empty"}), sorted(files[s])) for s, n in names.items() if "human" not in n}


def _select(seqs):
    """The subset and its split: all animal sequences + random empty sequences up to N_EMPTY_IMAGES images. Returns {split: [(file, label, seq_id)]}."""
    animal = sorted(s for s, (l, _) in seqs.items() if l == 1)
    empty = sorted(s for s, (l, _) in seqs.items() if l == 0)
    keep, n = [], 0
    for s in [empty[i] for i in np.random.default_rng(0).permutation(len(empty))]:
        if n >= N_EMPTY_IMAGES:
            break
        keep.append(s); n += len(seqs[s][1])
    out = {s: [] for s in fetch.SPLITS}
    for group in (animal, sorted(keep)):
        p = [group[i] for i in np.random.default_rng(0).permutation(len(group))]
        a, b = int(FRACTIONS[0] * len(p)), int((FRACTIONS[0] + FRACTIONS[1]) * len(p))
        for split, part in zip(fetch.SPLITS, (p[:a], p[a:b], p[b:])):
            out[split] += [(f, seqs[s][0], s) for s in part for f in seqs[s][1]]
    rng = np.random.default_rng(1)                       # mix animals and empties once, so any prefix (max_items) is representative
    return {s: [items[i] for i in rng.permutation(len(items))] for s, items in out.items()}


def preprocess(jpeg_bytes, size=SIZE):
    """JPEG bytes -> (3, size, size) uint8: centre square crop of the full frame, area downscale."""
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg_bytes))
    im.draft("RGB", (4 * size, 4 * size))          # let the JPEG decoder skip detail we throw away anyway (exact DCT-domain downscale)
    im = im.convert("RGB")
    w, h = im.size
    c = min(w, h)
    im = im.crop(((w - c) // 2, (h - c) // 2, (w - c) // 2 + c, (h - c) // 2 + c)).resize((size, size), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.uint8).transpose(2, 0, 1)


def build(path, workers=16):
    """Rebuild karoo.npz from LILA: download the metadata, pick the subset, stream every image through `preprocess`."""
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor
    meta_zip = fetch.chit_dir(CHIT) / "SnapshotKaroo_S1_v1.0.json.zip"
    if not meta_zip.exists() and not fetch.download(META_URL, meta_zip, "Snapshot Karoo metadata"):
        raise RuntimeError("could not download the Snapshot Karoo metadata from LILA")
    with zipfile.ZipFile(meta_zip) as z:
        meta = json.loads(z.read(z.namelist()[0]))
    plan = _select(_sequences(meta))

    def get(f):
        for _ in range(5):
            try:
                req = urllib.request.Request(IMAGE_URL + f, headers={"User-Agent": "aims-fastml-course"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    return preprocess(r.read())
            except Exception:
                pass
        return None

    import shutil
    cache = fetch.chit_dir(CHIT) / "build_cache"       # every 1,000 images are saved here, so an interrupted build resumes where it stopped
    cache.mkdir(exist_ok=True)
    out, CHUNK = {}, 1000
    for split, items in plan.items():
        X, ok = [], []
        for k in range(0, len(items), CHUNK):
            part = cache / f"{split}_{k // CHUNK:03d}.npz"
            if not part.exists():
                with ThreadPoolExecutor(workers) as ex:
                    ims = list(ex.map(get, [f for f, _, _ in items[k:k + CHUNK]]))
                np.savez(part, X=np.stack([np.zeros((3, SIZE, SIZE), np.uint8) if x is None else x for x in ims]),
                         ok=np.array([x is not None for x in ims]))
                print(f"  {split}: {min(k + CHUNK, len(items))}/{len(items)} images", flush=True)
            with np.load(part) as z:
                X.append(z["X"]); ok.append(z["ok"])
        X, ok = np.concatenate(X), np.concatenate(ok)
        print(f"  {split}: {ok.sum()}/{len(items)} images ({(~ok).sum()} failed to download)")
        out[f"X_{split}"] = X[ok]
        out[f"y_{split}"] = np.array([it[1] for it in items], np.int8)[ok]
        out[f"seq_{split}"] = np.array([it[2] for it in items])[ok]
    np.savez_compressed(path, **out)
    shutil.rmtree(cache)


def load(split="train", max_items=None, grayscale=False, size=SIZE):
    """Returns (X, y). X: (n, 3 or 1, size, size) uint8 (divide by 255 yourself); y: (n,) int64, 1 = animal, 0 = empty.
    `size` <= 96 downscales with area averaging; `grayscale=True` converts with the ITU-R 601 luma weights (as PIL's "L" mode)."""
    fetch.check(split)
    if not 1 <= size <= SIZE:
        raise ValueError(f"size must be between 1 and {SIZE}: the stored images are {SIZE} x {SIZE}")
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or 2_000
        y = (rng.random(n) < 0.35).astype(np.int64)
        X = rng.integers(0, 256, (n, 3, SIZE, SIZE), dtype=np.uint8)
    else:
        with np.load(fetch.get_npz(CHIT, NAME, KEYS, build)) as z:
            X, y = z[f"X_{split}"], z[f"y_{split}"].astype(np.int64)
    X, y = fetch.take(split, (X, y), max_items)
    if size != SIZE:
        from PIL import Image
        X = np.stack([np.asarray(Image.fromarray(x.transpose(1, 2, 0)).resize((size, size), Image.Resampling.BOX)).transpose(2, 0, 1) for x in X])
    if grayscale:
        X = np.round(np.tensordot(np.array([0.299, 0.587, 0.114]), X.astype(np.float32), axes=([0], [1])))[:, None].astype(np.uint8)
    return np.ascontiguousarray(X), y


def recall_at_empties_kept(score, y, kept=0.05):
    """`score`: higher = more likely an animal. Sets the threshold so that a fraction `kept` of the empty images pass,
    and returns the fraction of animal images that pass (recall)."""
    score, y = np.asarray(score), np.asarray(y)
    thr = np.quantile(score[y == 0], 1 - kept)
    return float(np.mean(score[y == 1] > thr))
