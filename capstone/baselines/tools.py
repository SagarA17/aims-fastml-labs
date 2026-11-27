"""Small helpers shared by the capstone baseline notebooks. Tools, not exercises: read them if you're curious, change them if you need to.

    from capstone.baselines.tools import fit, predict, to_onnx, ort_session, ort_int8, ort_predict, ort_latency, mcu_latency, energy_uj, report
"""
import json, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

FAST = os.environ.get("FASTML_FAST") == "1"      # set by tools/validate_chits.py --fast: fewer items and epochs, for a quick check only


def _device(device):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def fit(model, X, y, epochs=5, bs=256, lr=3e-3, loss_fn=None, device=None, class_weight=None, seed=0, verbose=True):
    """Adam + cosine schedule. X, y: numpy arrays or tensors (kept on the CPU, moved to `device` batch by batch; default: the GPU if there is one).
    loss_fn(output, y_batch) overrides cross-entropy. class_weight: per-class weights for cross-entropy (unbalanced data).
    Returns the model on the CPU, in eval mode."""
    torch.manual_seed(seed)
    device = _device(device)
    X, y = torch.as_tensor(X), (None if y is None else torch.as_tensor(y))
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(X) / bs))
    w = None if class_weight is None else torch.as_tensor(class_weight, dtype=torch.float32, device=device)
    for ep in range(epochs):
        t0, total = time.time(), 0.0
        for j in torch.randperm(len(X)).split(bs):
            xb = X[j].to(device).float()
            yb = None if y is None else y[j].to(device)
            out = model(xb)
            loss = loss_fn(out, yb) if loss_fn else F.cross_entropy(out, yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item() * len(j)
        if verbose:
            print(f"  epoch {ep + 1}/{epochs}: loss {total / len(X):.4f} ({time.time() - t0:.0f} s)")
    return model.cpu().eval()


@torch.inference_mode()
def predict(model, X, bs=1024, device=None):
    """Model outputs on X (numpy), as a numpy array (computed on the GPU if there is one; the model is left on the CPU)."""
    device = _device(device)
    model = model.to(device).eval()
    X = torch.as_tensor(X)
    out = torch.cat([model(X[i:i + bs].to(device).float()).float().cpu() for i in range(0, len(X), bs)]).numpy()
    model.cpu()
    return out


def to_onnx(model, example, path):
    """Export with a dynamic batch dimension. `example`: one input with its batch dimension, e.g. X[:1]."""
    model = model.cpu().eval()
    torch.onnx.export(model, torch.as_tensor(example).float(), path, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17, dynamo=False)
    return path


def ort_session(path, threads=1):
    """ONNX Runtime on the CPU with `threads` threads (1 = the EDGE-CPU rule)."""
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads, o.inter_op_num_threads = threads, 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def ort_int8(fp32_path, calib_X, int8_path, n_calib=512):
    """Static int8 quantization (weights AND activations, QDQ format, per tensor) with ONNX Runtime, calibrated on `calib_X`.
    This is the int8 model whose accuracy an MCU chit reports."""
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(np.asarray(calib_X[:n_calib], np.float32)[i:i + 1] for i in range(min(n_calib, len(calib_X))))
        def get_next(self):
            x = next(self.it, None)
            return None if x is None else {"x": x}

    pre = int8_path + ".pre.onnx"
    quant_pre_process(fp32_path, pre)
    quantize_static(pre, int8_path, Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
    os.remove(pre)
    return int8_path


def ort_predict(sess, X, bs=256):
    X = np.asarray(X, np.float32)
    return np.concatenate([sess.run(None, {"x": X[i:i + bs]})[0] for i in range(0, len(X), bs)])


def ort_latency(sess, x):
    """Batch-1 latency of an ONNX Runtime session with common/timing.py: {'median_us', 'p99_us'}. x: ONE sample with its batch dimension."""
    from common.timing import measure
    x = np.ascontiguousarray(np.asarray(x, np.float32))
    return measure(lambda: sess.run(None, {"x": x}))


def mcu_latency(macs, mac_rate):
    """MCU targets are counted, not measured: latency [s] = MACs / MAC rate."""
    return macs / mac_rate


# Energy per operation for the MCU energy budget, as on chit 6's card (Horowitz, ISSCC 2014, 45 nm; int8 everywhere).
E_MUL8, E_ADD8 = 0.2e-12, 0.03e-12          # per int8 multiply / add
E_FLASH_BYTE = 160e-12                       # weights from flash, at the DRAM figure: 640 pJ per 32 bits
E_SRAM_BYTE = 1.25e-12                       # activations in on-chip SRAM: 5 pJ per 32 bits


def energy_uj(p):
    """Energy per inference [µJ] for a `profile(...)` result, with the card's formula:
    MACs × (0.2 + 0.03) pJ + weight bytes × 160 pJ + activation bytes × 1.25 pJ, where activation bytes = the sum over layers of (input + output) bytes.
    Every weight is read once per inference."""
    act = sum(r["act_bytes"] for r in p["layers"])
    return (p["macs"] * (E_MUL8 + E_ADD8) + p["weight_bytes"] * E_FLASH_BYTE + act * E_SRAM_BYTE) * 1e6


def report(chit, n, input_shape, metric, prof, latency, licence, extra=None):
    """Print the one-line validation row that tools/validate_chits.py collects (and a readable version). Returns the row dict.
    n = (n_train, n_val, n_test); prof = common.budget.profile(...); latency = a string such as '412.3 / 455.0 µs (ORT, 1 thread)'."""
    row = dict(chit=chit, n="/".join(f"{k:,}" for k in n), input_shape=str(tuple(input_shape)), metric=metric,
               params=prof["params"], macs=prof["macs"], int8_weight_bytes=prof["weight_bytes"], peak_act_bytes=prof["peak_act_bytes"],
               latency=latency, licence=licence, fast=FAST, **(extra or {}))
    for k, v in row.items():
        print(f"  {k:<18} {v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"  {k:<18} {v}")
    print("VALIDATION_ROW " + json.dumps(row, ensure_ascii=False))
    return row
