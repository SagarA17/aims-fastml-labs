"""Count what a model costs: the tool behind the MCU and EDGE-CPU capstone targets (and a sanity check for every other target).

    from common.budget import profile, fits
    p = profile(model, input_shape=(1, 96, 96))          # one sample, without the batch dimension
    print(p)                                              # params, MACs, weight bytes, peak activation bytes, per-layer table
    fits(p, target)                                       # compare against a target dict from capstone/targets.md

Counting rules (the same as in lectures):
  * MACs: Linear = in × out; Conv1d/2d = out_elements × (in_channels / groups) × kernel_size; batched matmul (attention) is counted
    via `extra_macs` if you use torch.matmul directly. Biases, activations and pooling are ignored.
  * Weight bytes = params × bytes_per_weight (int8 → 1).
  * Peak activation bytes: walking the leaf modules in execution order, the largest (input + output) tensor size of any layer. That's what must live in
    SRAM at once for a sequential model running layer by layer. Skip connections keep extra tensors alive; add them by hand (`extra_act_bytes`).
"""
import torch
import torch.nn as nn

_COUNTED = (nn.Linear, nn.Conv1d, nn.Conv2d)


def _layer_macs(m, out):
    if isinstance(m, nn.Linear):
        return m.in_features * m.out_features * (out.numel() // out.shape[-1] if out.dim() > 1 else 1)
    if isinstance(m, (nn.Conv1d, nn.Conv2d)):
        k = 1
        for s in m.kernel_size:
            k *= s
        return out.numel() * (m.in_channels // m.groups) * k
    return 0


def profile(model, input_shape, bytes_per_weight=1, bytes_per_act=1, extra_macs=0, extra_act_bytes=0):
    """Profile ONE sample. input_shape excludes the batch dimension. Returns a dict, including a per-layer list."""
    rows = []

    def hook(m, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        o = out[0] if isinstance(out, tuple) else out
        rows.append(dict(layer=m.__class__.__name__, out_shape=tuple(o.shape[1:]), macs=_layer_macs(m, o[0:1]) if o.dim() > 0 else 0,
                         params=sum(p.numel() for p in m.parameters(recurse=False)),
                         act_bytes=(x[0:1].numel() + o[0:1].numel()) * bytes_per_act))

    leaves = [m for m in model.modules() if len(list(m.children())) == 0]
    hs = [m.register_forward_hook(hook) for m in leaves]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, *input_shape))
    for h in hs:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    macs = sum(r["macs"] for r in rows) + extra_macs
    return dict(params=params, macs=int(macs), weight_bytes=int(params * bytes_per_weight),
                peak_act_bytes=int(max(r["act_bytes"] for r in rows) + extra_act_bytes),
                input_bytes=int(torch.zeros(input_shape).numel() * bytes_per_act), layers=rows)


def fits(p, target, verbose=True):
    """target: dict with any of max_weight_bytes, max_act_bytes, max_macs, mac_rate (MAC/s) + max_latency_s. Returns (ok, report)."""
    checks = []
    if "max_weight_bytes" in target:
        checks.append(("weights (flash)", p["weight_bytes"], target["max_weight_bytes"]))
    if "max_act_bytes" in target:
        checks.append(("peak activations (SRAM)", p["peak_act_bytes"], target["max_act_bytes"]))
    if "max_macs" in target:
        checks.append(("MACs per inference", p["macs"], target["max_macs"]))
    if "mac_rate" in target and "max_latency_s" in target:
        checks.append(("latency estimate [s]", p["macs"] / target["mac_rate"], target["max_latency_s"]))
    ok = all(v <= lim for _, v, lim in checks)
    report = "\n".join(f"  {'OK ' if v <= lim else 'OVER'} {name:<26} {v:>14,.4g} / {lim:,.4g}  ({100 * v / lim:5.1f}%)" for name, v, lim in checks)
    if verbose:
        print(("FITS" if ok else "DOES NOT FIT") + "\n" + report)
    return ok, report
