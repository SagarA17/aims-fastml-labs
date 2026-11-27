# Capstone: A fast-ML system under a real budget
**AIMS Edge ML for Science · Teams of 2–3 · Out: Day 5 · Presentations: Day 10 · 40% of the course (provisional)**

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
2. **Choose ≥ 3 candidate architectures and cost them *before* training.** Use `common/budget.py` and the Day 7 scaling laws, with at least one "simple" baseline
   (e.g. an MLP, or a non-ML threshold) and at least one architecture that matches the data's structure (CNN, Deep Sets, …). Predict which fit.
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
| **Day 7, 18:00** | **Checkpoint** (🟥 AI-off, 1 page): your budget in numbers + ≥ 3 candidates, each with counted cost and a predicted fit. Feedback the next morning. |
| **Day 8, 20:00** | FPGA teams: **synthesis round 1** (≤ 2 designs, via `make_submission`). Results back by Day 9 midday. |
| **Day 9, 20:00** | FPGA teams: **synthesis round 2** (≤ 2 designs). Results back by Day 10 morning. |
| **Day 10, 09:00** | Notebook due (executed, with outputs). |
| **Day 10** | Presentations: **8 min + 3 min questions** per team. |

## Presentation (Day 10)
Use 5–7 slides, in any tool. Structure: **the budget → the candidates and why → the trade-off front → prediction vs. reality → what you'd do next.**
Every team member presents part of the talk, and **each member answers at least one question**.
Lead with the budget, not the dataset: the audience has a different chit.

## Grading (100 points; detailed rubric on request)
| Component | Points | Notes |
|---|---|---|
| Checkpoint (Day 7) | 10 | Budget in numbers, ≥ 3 costed candidates, predicted fit. Graded on reasoning, not on whether the predictions come true. |
| Notebook | 55 | Budget analysis (10) · candidates costed before training (10) · trade-off front, with seeds/spread where it matters (15) · prediction vs. measurement/synthesis, misses explained (15) · honesty about what didn't work (5) |
| Presentation | 25 | Clarity of the budget story (10) · evidence for the claims (10) · timing and slides (5) |
| Individual | 10 | Your answer to a question, plus the team's contribution statement (one line per person, in the notebook) |

**The budgets are targets, not walls.** They were set from one baseline run each, not tested exhaustively. Aim for your final design to fit. If it misses, report by how much and why, and what you would try next: that is graded as highly as a design that fits.

## Rules
- 🟩 **AI-on** for code. 🟥 **AI-off** for the checkpoint, for the numbers in your budget analysis, and for the Q&A. You must be able to explain every line and every number you show.
- Use only your chit's data, with the **fixed splits** from its loader. Don't tune on the test split.
- FPGA teams: only designs built from the provided builders can be synthesized (see `targets.md`). You may *cost* anything.
- Cite the papers you build on. Comparing your result with the published reference design is encouraged.
