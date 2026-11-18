"""Package an FPGA design for instructor synthesis (capstone teams with an FPGA target).

A submission is one file: the model weights, its per-layer bit configuration, the hls4ml settings, and YOUR PREDICTIONS for the
synthesis results. Before writing anything, `make_submission` converts the model and checks that the hls4ml emulation matches PyTorch bit for bit,
so the instructor never spends synthesis time on a broken design.
"""
import os, re, shutil, time
import numpy as np
import torch

from .compress import SIZES, accuracy
from .hls import to_hls, check_bit_exact

REQUIRED_PREDICTIONS = ("dsp", "lut", "latency_cycles", "ii")


def make_submission(model, cfg, team, name, reuse_factor, predicted, X_check, y_check=None, sizes=SIZES,
                    io_type="io_parallel", clock_period=5.0, out_dir="submissions"):
    """model: a qat_mlp(cfg, sizes) model (pruning masks removed or not; both work).
    predicted: dict with your predictions for dsp, lut, latency_cycles, ii (and optionally anything else).
    Returns the path of the written submission file."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", team) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", name):
        raise ValueError("team and name: letters, digits, '-' or '_' only")
    missing = [k for k in REQUIRED_PREDICTIONS if k not in predicted]
    if missing:
        raise ValueError(f"predictions missing: {missing}. Predict before you submit!")

    import torch.nn.utils.prune as prune
    for m in model.modules():                       # make any pruning permanent
        if hasattr(m, "weight_orig"):
            prune.remove(m, "weight")
    model.eval()

    tmp = os.path.join(out_dir, f"_check_{team}_{name}")
    hm, _ = to_hls(model, output_dir=tmp, reuse_factor=reuse_factor, io_type=io_type, clock_period=clock_period)
    diff, agree = check_bit_exact(hm, model, np.asarray(X_check[:2000]))
    shutil.rmtree(tmp, ignore_errors=True)
    if diff != 0:
        raise RuntimeError(f"hls4ml emulation does not match PyTorch (max diff {diff}). Use qat_mlp from common/compress.py.")

    sub = dict(team=team, name=name, cfg=[[int(w), int(a)] for w, a in cfg], sizes=[int(s) for s in sizes], reuse_factor=int(reuse_factor),
               io_type=io_type, clock_period=float(clock_period), predicted={k: float(v) for k, v in predicted.items()},
               state_dict=model.state_dict(), created=time.strftime("%Y-%m-%d %H:%M:%S"),
               local_acc=None if y_check is None else accuracy(model, torch.as_tensor(X_check), torch.as_tensor(y_check)))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{team}__{name}.pt")
    torch.save(sub, path)
    print(f"bit-exact OK -> wrote {path}  (send this file to the instructor)")
    return path
