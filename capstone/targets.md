# Hardware targets
Every chit names one target. The numbers in angle brackets ⟨…⟩ are set on your **chit card**. This file explains what each target means and how you prove a design fits.

**Emulated targets.** Nobody in this course has an MCU or a Raspberry Pi on the desk, so MCU and EDGE-CPU are *specifications* you design against:
you count, and measure on your laptop with a stated correction. That's how a lot of real embedded work starts. Say it clearly in your talk.

---

## FPGA-L1: a Level-1 trigger FPGA
- **Spec:** a new input every 25 ns (II ≤ ⟨5⟩ cycles at 200 MHz), latency ≤ ⟨N⟩ cycles, ≤ ⟨D⟩ DSPs, ≤ ⟨L⟩ LUTs (C-synthesis estimates).
  The LUT limit is a rule on your card (a multiple of the baseline's synthesized LUTs); the number is given on Monday, after the instructor synthesizes the baseline.
- **How you evaluate:** convert with `common/hls.py` (bit-exact emulation on your laptop), estimate resources with `dense_summary` and the synthesis grid (Lab 3),
  then submit with `common/submit.py` for instructor **C-synthesis** (2 rounds, ≤ 2 designs each).
- **Synthesizable designs:** built from the provided Brevitas builder `qat_mlp` (any widths, so autoencoders too). Anything else you may cost on paper.
- **Typical levers:** bit width (DSP → LUT below ~10 bits), pruning, reuse factor (II ↔ resources), and width/depth.

## FPGA-ns: a qubit-control FPGA (RFSoC-class)
- **Spec:** latency ≤ ⟨50 ns⟩ (≈ 10 cycles at ~200 MHz), II = 1, ≤ ⟨L⟩ LUTs, ≤ ⟨D⟩ DSPs.
- **How you evaluate:** exactly as FPGA-L1. The tight latency means fully unrolled, shallow, low-precision designs.

## MCU: a microcontroller (Cortex-M4/M7-class, emulated)
- **Spec:** int8 weights ≤ ⟨W⟩ bytes (flash), peak activations ≤ ⟨A⟩ bytes (SRAM), latency ≤ ⟨T⟩ at an assumed ⟨R⟩ MAC/s.
  Energy (if your chit asks): use this table (Horowitz, 45 nm; int8 everywhere):

  | Operation | Energy |
  |---|---|
  | int8 multiply | 0.2 pJ |
  | int8 add | 0.03 pJ |
  | read 1 byte from on-chip SRAM (activations) | 1.25 pJ |
  | read 1 byte from flash (weights; the DRAM figure, 640 pJ per 32 bits) | 160 pJ |

  E ≈ MACs × (0.2 + 0.03) pJ + weight bytes × 160 pJ + activation bytes × 1.25 pJ, where *activation bytes* = the sum over layers of (input + output) bytes
  (the `act_bytes` column of `profile(...)["layers"]`). Count every weight as read once per inference, and include feature extraction in the MACs. State any other assumption you make.
- **How you evaluate:** `common/budget.py`: `profile(model, input_shape)` then `fits(p, target)`. Accuracy **must** be measured with int8 weights *and* activations
  (Lab 2 QAT, or ONNX Runtime static int8 from Lab 4). Count any preprocessing (e.g. MFCCs) in the MACs using `extra_macs`.
- **Why latency is counted, not measured.** A microcontroller has one simple core, no operating system and no caches to speak of, and its int8 kernels run the MACs
  one after another at an essentially fixed rate. So its latency is, to a good approximation, **MACs ÷ MAC rate**: that is the *emulation expectation*, and it's what your budget is checked against.
- **Sanity check (required): measure as well.** Run each candidate with ONNX Runtime, int8, on **1 thread** of Colab's CPU, timed with `common/timing.py` (Lab 4 shows how).
  The absolute numbers will be far *faster* than the MCU spec (a server core is not a microcontroller); what matters is whether your candidates **rank** the same way as their MAC counts.
  If a model with fewer MACs runs slower, find out why (many small layers? an operation that isn't a MAC, like the MFCC or pooling?) and say so: that's exactly where "count the MACs" stops being the whole story.
- **Typical levers:** depthwise-separable convs, input resolution, early downsampling (peak activations!), and width.

## EDGE-CPU: a Raspberry-Pi-class CPU (emulated)
- **Spec:** p99 latency ≤ ⟨T⟩ ms on **1 thread**, model file ≤ ⟨M⟩ MB.
- **How you evaluate:** ONNX Runtime with `intra_op_num_threads=1` (Lab 4), timed with the Lab 4 helper at batch 1, measured on your laptop, then multiplied by a slow-down factor of ⟨k⟩ (on your card).
  Report the laptop number *and* the scaled one.
- **Typical levers:** ONNX Runtime + int8, input size, architecture, and width.

## GPU-RT: a real-time GPU
- **Spec:** one sample at a time (**batch 1**, no batching), p99 latency ≤ ⟨T⟩ ms per sample on Colab's T4, **including** the copy of the input to the GPU and of the result back;
  GPU memory ≤ ⟨M⟩ GB.
- **How you evaluate:** use the Lab 4 timing helper (it handles warm-up and repetition for you) and report the median and the p99.
- **Typical levers:** architecture (depth, width, early downsampling), input size, fp16, `torch.compile` and CUDA graphs, ONNX Runtime,
  and *structured* pruning (fewer channels). All from Lab 4.
