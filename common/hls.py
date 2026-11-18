"""Bridge: Brevitas QAT model (built with common.compress.qat_mlp) -> QONNX -> hls4ml -> compiled C++ emulation.

Two details are handled here for you, so the hls4ml emulation matches Brevitas BIT-EXACTLY:
  * Biases are quantized in Brevitas (qat_mlp does this). With float biases, hls4ml has to choose a fixed-point type for them, and outputs drift.
  * `hls_predict` feeds hls4ml inputs that are already on the input quantizer's grid. A raw float input would first be cast to
    hls4ml's default type and then rounded again by the Quant layer (double rounding), changing a small fraction of results.
No Vivado/Vitis is needed for conversion or emulation. Synthesis (resources, latency) is a separate step.
"""
import os, io, contextlib, warnings
import numpy as np
import torch


def to_hls(model, output_dir="hls_prj", in_features=16, reuse_factor=1, io_type="io_parallel",
           part="xcu250-figd2104-2L-e", backend="Vitis", default_precision="fixed<16,6>", compile=True, verbose=False,
           clock_period=5.0, project_name="myproject"):
    """Convert a Brevitas model to an hls4ml model. Returns (hls_model, hls_config)."""
    warnings.filterwarnings("ignore")
    from brevitas.export import export_qonnx
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.util.cleanup import cleanup_model
    from qonnx.transformation.channels_last import ConvertToChannelsLastAndClean
    from qonnx.transformation.gemm_to_matmul import GemmToMatMul
    import hls4ml

    model = model.eval()
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model_qonnx.onnx")
    export_qonnx(model, torch.zeros(1, in_features), export_path=onnx_path)
    m = cleanup_model(ModelWrapper(onnx_path))
    m = cleanup_model(m.transform(ConvertToChannelsLastAndClean()).transform(GemmToMatMul()))
    with (contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())):
        cfg = hls4ml.utils.config.config_from_onnx_model(m, granularity="name", backend=backend,
                                                         default_precision=default_precision, default_reuse_factor=reuse_factor)
        hm = hls4ml.converters.convert_from_onnx_model(m, output_dir=output_dir, io_type=io_type, backend=backend,
                                                       hls_config=cfg, part=part, clock_period=clock_period,
                                                       project_name=project_name)
        if compile:
            hm.compile()
    return hm, cfg


def quantize_inputs(model, X):
    """Run only the model's input quantizer (the first module), giving values on the exact grid the FPGA sees."""
    with torch.no_grad():
        out = model[0](torch.as_tensor(X))
        out = out.value if hasattr(out, "value") else out
    return out.numpy().astype(np.float32)


def hls_predict(hls_model, model, X):
    """hls4ml C++ emulation on inputs pre-quantized by the Brevitas input quantizer."""
    xq = np.ascontiguousarray(quantize_inputs(model, X))
    return np.asarray(hls_model.predict(xq)).reshape(len(xq), -1)


def check_bit_exact(hls_model, model, X):
    """Returns (max_abs_diff, prediction_agreement) between Brevitas and the hls4ml emulation."""
    with torch.no_grad():
        yb = model(torch.as_tensor(X)).numpy()
    yh = hls_predict(hls_model, model, X)
    return float(np.abs(yb - yh).max()), float((yb.argmax(1) == yh.argmax(1)).mean())


def layer_types(hls_model):
    """{layer name: {variable: ap type}} as chosen by hls4ml: weights, biases, results, accumulators."""
    out = {}
    for layer in hls_model.get_layers():
        d = {}
        for k, v in layer.attributes.items():
            if hasattr(v, "precision") and hasattr(v, "type"):
                d[k] = str(v.precision)
            elif k.endswith("_t") and hasattr(v, "precision"):
                d[k] = str(v.precision)
        try:
            d["result"] = str(layer.get_output_variable().type.precision)
        except Exception:
            pass
        acc = layer.get_attr("accum_t")
        if acc is not None:
            d["accum"] = str(acc.precision)
        out[layer.name] = d
    return out


def dense_summary(hls_model):
    """Per Dense layer: MACs, zero weights (pruned or quantized to 0), the multipliers left to build, and the reuse factor.
    `mults_needed` = non-zero weights / RF is a first-order estimate of multiplier count. Synthesis decides DSP vs. LUT."""
    rows = []
    for L in hls_model.get_layers():
        if L.class_name != "Dense":
            continue
        n_in, n_out, rf = L.get_attr("n_in"), L.get_attr("n_out"), L.get_attr("reuse_factor")
        nz = getattr(L.weights.get("weight"), "nzeros", 0)
        rows.append(dict(layer=L.name, macs=int(n_in * n_out), zero_weights=int(nz), nonzero=int(n_in * n_out - nz), reuse_factor=int(rf),
                         mults_needed=int(-(-(n_in * n_out - nz) // rf)),
                         weight_t=str(L.get_attr("weight_t").precision), accum_t=str(L.get_attr("accum_t").precision)))
    return rows

