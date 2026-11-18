"""Shared helpers for compression work (A2, Lab 4, capstone). These are tools, not exercises.

Quantized models use power-of-two ("FixedPoint") quantizers for weights, activations AND biases. That is exactly what maps onto
hls4ml's ap_fixed types, so any model built with `qat_mlp` can be converted to an FPGA design bit-exactly (Lab 4).
"""
import copy, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.nn.utils.prune as prune

SIZES = (16, 64, 32, 32, 5)


def mlp(sizes=SIZES):
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def qat_mlp(cfg, sizes=SIZES):
    """Brevitas MLP. cfg[i] = (w_bits, a_bits) for Linear layer i, where a_bits is the bit width of that layer's INPUT
    (the input quantizer for layer 0, the preceding ReLU's quantizer otherwise). Same convention as Lab 2."""
    import brevitas.nn as qnn
    from brevitas.quant import (Int8WeightPerTensorFixedPoint, Int8ActPerTensorFixedPoint, Uint8ActPerTensorFixedPoint,
                                Int8BiasPerTensorFixedPointInternalScaling)
    n = len(sizes) - 1
    assert len(cfg) == n, f"need {n} (w_bits, a_bits) pairs"
    layers = [qnn.QuantIdentity(act_quant=Int8ActPerTensorFixedPoint, bit_width=cfg[0][1])]
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        wb = cfg[i][0]
        layers.append(qnn.QuantLinear(a, b, bias=True, weight_quant=Int8WeightPerTensorFixedPoint, weight_bit_width=wb,
                                      bias_quant=Int8BiasPerTensorFixedPointInternalScaling, bias_bit_width=max(12, 2 * wb)))
        if i < n - 1:
            layers.append(qnn.QuantReLU(act_quant=Uint8ActPerTensorFixedPoint, bit_width=cfg[i + 1][1]))
    return nn.Sequential(*layers)


def train(model, X, y, epochs=4, bs=1024, lr=3e-3, loss_fn=None, seed=None):
    """Adam + cosine schedule. loss_fn(logits, idx) overrides cross-entropy (idx = batch indices into X)."""
    if seed is not None:
        torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * math.ceil(len(X) / bs))
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            out = model(X[j])
            loss = F.cross_entropy(out, y[j]) if loss_fn is None else loss_fn(out, j)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.inference_mode()
def logits(model, X, bs=65536):
    model.eval()
    return torch.cat([model(X[i:i + bs]) for i in range(0, len(X), bs)])


def accuracy(model, X, y):
    return (logits(model, X).argmax(1) == y).float().mean().item()


def agreement(model_a, model_b, X):
    """Fraction of inputs on which two models predict the same class (a.k.a. fidelity)."""
    return (logits(model_a, X).argmax(1) == logits(model_b, X).argmax(1)).float().mean().item()


def bkg_rejection(model, X, y, sig=4, eff=0.5):
    """One-vs-rest tagger for class `sig` (default: top). Returns 1/FPR at the given signal efficiency."""
    p = torch.softmax(logits(model, X), 1)[:, sig].numpy()
    s, b = p[y.numpy() == sig], p[y.numpy() != sig]
    cut = np.quantile(s, 1 - eff)
    fpr = (b >= cut).mean()
    return float("inf") if fpr == 0 else 1.0 / fpr


def linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def prune_global(model, amount):
    """Global L1 (magnitude) unstructured pruning of all Linear weights. The mask stays fixed during later training."""
    prune.global_unstructured([(m, "weight") for m in linears(model)], pruning_method=prune.L1Unstructured, amount=amount)
    return model


def sparsity_per_layer(model):
    return [float((m.weight == 0).float().mean()) for m in linears(model)]


def split_by_feature(X, y, feature_names, name="j_multiplicity", q=0.8, threshold=None):
    """Split into (in-range, out-of-range) by a feature's value. The threshold defaults to the q-quantile of X.
    Returns (X_in, y_in, X_out, y_out, threshold)."""
    if name in list(feature_names):
        k = list(feature_names).index(name)
    else:
        k = X.shape[1] - 1
        print(f"[split_by_feature] '{name}' not found (synthetic data?); using column {k}")
    v = X[:, k]
    thr = np.quantile(v, q) if threshold is None else threshold
    m = v <= thr
    return X[m], y[m], X[~m], y[~m], thr


def snapshot(model):
    """Deep-copy a model, including one that is mid-pruning. (A pruned weight is recomputed from weight_orig * mask on every forward, and
    after a training step it carries autograd history, which deepcopy refuses. A no-grad forward refreshes it first.)"""
    with torch.no_grad():
        model.eval()(torch.zeros(1, linears(model)[0].in_features))
    return copy.deepcopy(model)
