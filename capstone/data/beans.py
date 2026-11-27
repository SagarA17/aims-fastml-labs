"""Reserve R1: a bean-leaf disease classifier for a phone-class CPU (EDGE-CPU). iBean, Makerere AI Lab, Uganda.

    from capstone.data.beans import load, CLASSES
    X, y = load("train")            # X: (1034, 3, 128, 128) uint8 RGB, y: (1034,) int64 in 0..2 = CLASSES
    X, y = load("train", size=224)  # any size up to the stored 500 x 500: input size is a latency lever

**Source.** AI Lab, Makerere University, *iBean* (2020): https://github.com/AI-Lab-Makerere/ibean ; files from
https://huggingface.co/datasets/AI-Lab-Makerere/beans (three parquet files). Smartphone photos of bean leaves taken in Ugandan fields and annotated
by experts from the National Crops Resources Research Institute (NaCRRI). **Licence:** **MIT** (the GitHub repository's LICENSE and the Hugging Face card agree).
**Cite:** Makerere AI Lab, *Bean disease dataset* (2020), https://github.com/AI-Lab-Makerere/ibean/

**Classes.** 0 angular leaf spot, 1 bean rust, 2 healthy (roughly balanced).

**Splits (fixed).** The dataset's own train / validation / test split (1,034 / 133 / 128 images), unchanged. The authors made it; we don't know
whether photos of the same plant appear in more than one split, so treat the test accuracy as optimistic.

**Preprocessing.** The images are 500 × 500 JPEGs. `load(size=s)` decodes them and resizes the whole image to s × s with area averaging (no crop).
That is the only preprocessing; normalisation is up to you.

**Download.** ~0.15 GB of parquet from Hugging Face (fast), no course release needed.
"""
import io
import numpy as np
from capstone.data import fetch

CHIT = "beans"
HF = "https://huggingface.co/datasets/AI-Lab-Makerere/beans/resolve/main/data"
FILES = {"train": "train-00000-of-00001.parquet", "val": "validation-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}
CLASSES = ["angular_leaf_spot", "bean_rust", "healthy"]


def _parquet(split):
    path = fetch.chit_dir(CHIT) / FILES[split]
    if not path.exists() and not fetch.download(f"{HF}/{FILES[split]}", path, f"beans {split} split from Hugging Face"):
        raise RuntimeError(f"could not download {FILES[split]}")
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pydict()


def load(split="train", max_items=None, size=128):
    """Returns (X, y): X (n, 3, size, size) uint8 RGB (divide by 255 yourself), y (n,) int64 in 0..2 (CLASSES)."""
    fetch.check(split)
    if fetch.synthetic():
        rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
        n = max_items or {"train": 1034, "val": 133, "test": 128}[split]
        return rng.integers(0, 256, (n, 3, size, size), dtype=np.uint8), rng.integers(0, 3, n).astype(np.int64)
    from PIL import Image
    t = _parquet(split)
    imgs, labels = t["image"], t["labels"]
    if max_items:
        imgs, labels = imgs[:max_items], labels[:max_items]
    X = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB").resize((size, size), Image.Resampling.BOX)).transpose(2, 0, 1)
                  for im in imgs])
    return X, np.asarray(labels, np.int64)
