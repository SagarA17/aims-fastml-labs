"""Batch-1 latency, measured the same way everywhere (Lab 4, the capstone CPU and GPU targets, instructor validation).

    from common.timing import measure, cpu_call, gpu_call
    r = measure(cpu_call(model, x))                  # x: ONE sample with its batch dimension, e.g. shape (1, 30, 4)
    r = measure(gpu_call(model_on_gpu, x), cuda=True) # includes the copy to the GPU and of the result back
    print(r)                                          # {'median_us': ..., 'p99_us': ...}

`fn` is any zero-argument callable that does one complete inference, e.g. `lambda: session.run(None, {"x": x_np})` for ONNX Runtime.
Repetition and warm-up are handled here, so the numbers are stable on a shared machine: report what this returns.
"""
import time
import numpy as np
import torch


def measure(fn, cuda=False, n=1000, warmup=100, repeats=5):
    """Time `fn()` one call at a time. Returns {'median_us', 'p99_us'}: the median over `repeats` rounds of each round's median and p99."""
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    meds, p99s = [], []
    t = np.empty(n)
    for _ in range(repeats):
        for i in range(n):
            t0 = time.perf_counter()
            fn()
            if cuda:
                torch.cuda.synchronize()
            t[i] = time.perf_counter() - t0
        meds.append(np.median(t)); p99s.append(np.percentile(t, 99))
    return {"median_us": round(float(np.median(meds)) * 1e6, 1), "p99_us": round(float(np.median(p99s)) * 1e6, 1)}


def cpu_call(model, x):
    """One PyTorch inference on the CPU, without autograd."""
    model.eval()
    def fn():
        with torch.inference_mode():
            return model(x)
    return fn


def gpu_call(model, x):
    """One PyTorch inference on the GPU, counted end to end: copy the input (on the CPU) to the GPU, run, copy the result back."""
    model.eval()
    def fn():
        with torch.inference_mode():
            return model(x.to("cuda")).cpu()
    return fn
