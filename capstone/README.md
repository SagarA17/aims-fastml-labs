# Capstone: A fast-ML system under a real budget
**AIMS Edge ML for Science · Teams of 4–5 · Out: Day 5 · Presentations: Day 10 · 40% of the course (provisional)**

Your team draws a **chit**: a dataset, a hardware target, and a budget. Your job is to build the best model you can **that provably fits**,
and to explain *why* it's the best you could do. There is no single right answer, and teams with different chits are not competing with each other.
You are graded on how well you **reason, predict and measure**, not on the final accuracy.

- `chits.md`: the six chits, with their data, targets and budgets. Two teams may draw the same chit.
- `baselines/`: one baseline notebook per chit: data loading, a simple model that roughly fits, and every number the budget asks for. Start from it.
- `targets.md`: what each hardware target means and how you evaluate a design against it.
- `TEMPLATE.ipynb`: the notebook structure your submission must follow.
- Tools you already have: `common/budget.py` (counting), Lab 2/A2 (compression), Lab 3 + `common/hls.py` + `common/submit.py` (FPGA), Lab 4 (profiling, ONNX Runtime, `torch.compile`, CUDA graphs: real-time inference at batch 1 on CPU and GPU).

---

## What you must do
1. **Understand the budget** (🟥 AI-off). Turn the chit's requirement into numbers: operations per inference, memory, latency and deadline, and the arithmetic
   intensity you can afford. Where on the roofline (Day 1) or in the resource table (Day 5) does your target live?
2. **On top of your chit's baseline, choose at least 2 candidate architectures of your own, and cost them *before* training.** At least one of them must match
   the structure of your data (e.g. a CNN for images, spectrograms or waveforms). Count all of them (the baseline too) with `common/budget.py` and the Day 7 scaling laws,
   and predict which fit. You are welcome to try more.
3. **Train, compress, and map the trade-off front.** Quantization, pruning, distillation and width/depth: whatever fits your target. Plot your metric against cost
   (in the unit your target cares about: bytes, MACs, latency, or DSP/LUT).
4. **Predict, then measure (or synthesize).** For your final 1–2 designs, write down the predicted cost, then measure it (CPU/GPU), count it (MCU/EDGE), or have it synthesized
   (FPGA, 2 rounds; see `targets.md`). Explain every miss larger than ~30%.
5. **Be honest about what didn't work.** A clearly explained dead end is worth marks; hiding it is not.

## Timeline
| When | What |
|---|---|
| **Day 5**, end of lecture | Teams formed, chits drawn. Read your chit, `targets.md`, and the paper(s) on your chit card. |
| **Day 6** | Lab 3 (hls4ml: the FPGA tools). Start the data loader and baseline. |
| **Day 8** | **Checkpoint**: your budget in numbers + the baseline and your ≥ 2 candidates, each with counted cost and a predicted fit. |
| **Day 8** | FPGA teams: **synthesis round 1** (≤ 2 designs, via `make_submission`). |
| **Day 9** | FPGA teams: **synthesis round 2** (≤ 2 designs). |
| **Day 10** | Notebook due (executed, with outputs). |
| **Day 10** | Team Presentations |

## Presentation (Day 10)
Use 5–7 slides, in any tool. Structure: **the budget → the candidates and why → the trade-off front → prediction vs. reality → what you'd do next.**
Every team member presents part of the talk, and **each member answers at least one question**.
Lead with the budget, not the dataset: the audience has a different chit.

**The budgets are targets, not walls.** They were set from one baseline run each, not tested exhaustively. Aim for your final design to fit. If it misses, report by how much and why, and what you would try next.

## Rules
- 🟩 **AI-on** for code. 🟥 **AI-off** for the model choice, for the numbers in your budget analysis, and for the Q&A. You must be able to explain every line and every number you show. Please do not use AI for creative choices!
- Use only your chit's data, with the **fixed splits** from its loader. Don't tune on the test split.
- FPGA teams: only designs built from the provided builders can be synthesized (see `targets.md`). You may *cost* anything.
- Cite the papers you build on. Comparing your result with the published reference design is encouraged.
