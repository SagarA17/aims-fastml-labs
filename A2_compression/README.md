# Assignment A2: Let HGQ choose the bit widths
**Out:** Friday 27 November · **Due:** Monday 30 November, 22:00 · **Effort:** ~6 hours, of which **about 1.5 hours is training** (it runs without you, but start early: do not leave it for the last evening) · **Weight:** 20% of the course grade

Open `A2.ipynb` in Colab (see the course README), save a copy in Drive, set the runtime to **T4 GPU**, and work through it.

Build the best **W-boson tagger** for about **50k BOPs** with **High-Granularity Quantization (HGQ)**, which learns a separate bit width for every weight and every input.
Why the W tagger: in Lab 2 you saw that W and Z jets lose accuracy first when the bits are reduced (they are often confused with each other), more than top jets.
You will compare HGQ with the Lab 2 method (the same bit width everywhere). Every part starts with a complete worked example, which you then extend.

| Part | | Weight |
|---|---|---|
| 1. Worked example: train one HGQ model (5 epochs with β = 0, then β increases slowly) and plot W rejection against EBOPs after every epoch | 🟩 | 5% |
| 2. Choosing and fine-tuning a model: the model just below 100k / 50k / 10k EBOPs, bit widths frozen, 5 more epochs; 3 seeds | 🟩 code · 🟥 answers | 25% |
| 3. HGQ vs. uniform bit widths: Brevitas models with the same bits everywhere, 3 seeds each, on the same plot | 🟩 code · 🟥 answers | 20% |
| 4. What HGQ removes: jet features, neurons and bits at 100k, 50k, 10k and 1k EBOPs | 🟩 code · 🟥 answers | 25% |
| 5. Your own improvement at about 50k: one idea of your own, 3 seeds | 🟩 code · 🟥 answers | 15% |
| Fair comparisons and clear plots: same data and number of epochs, 3 seeds, readable plots, complete PDF | | 10% |

**Submit one PDF**
1. Run every cell, so all outputs and plots are visible; write your answers in the *Your answer* cells.
2. **File → Print → Save as PDF**, and check that your plots and numbers appear.
3. Name it `A2_Surname_Firstname.pdf`.
4. Email it by **Monday 30 November, 22:00** to **aymantarig@aims.ac.za** and **hassanh@aims.ac.za**, CC **sagar@slac.stanford.edu**,
   with the subject `[EdgeML] A2 – Surname Firstname`.
