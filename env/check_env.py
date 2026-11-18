"""Run `python check_env.py` from the env/ folder before Day 2. It checks that the course packages import,
downloads and verifies the jet dataset (REQUIRED, one time, needs internet), and reports CPU info.
Use `--no-data` to skip the dataset step."""
import importlib, platform, os, sys
from pathlib import Path
pkgs = ["numpy", "scipy", "matplotlib", "sklearn", "torch", "brevitas", "onnx", "onnxruntime", "onnxoptimizer", "qonnx", "hls4ml"]
ok = True
for p in pkgs:
    try:
        m = importlib.import_module(p); print(f"[ok]   {p:12s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        ok = False; print(f"[FAIL] {p:12s} {e}")
print(f"\nPython {platform.python_version()} | {platform.system()} {platform.machine()} | {os.cpu_count()} logical cores")

if "--no-data" not in sys.argv:
    print("\nDataset (required before Day 2):")
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from common.jets import check_jets
        ok = check_jets() and ok
    except Exception as e:
        ok = False; print(f"[FAIL] jet dataset: {e}")
print("\nAll good." if ok else "\nFix the failures above, or bring your laptop to the setup clinic.")
