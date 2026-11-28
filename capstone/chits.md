# The chits
Each team draws one. Two teams may draw the same chit. Each card lists the paper(s) to start from.
Data loaders live in `capstone/data/<chit>.py` and expose `load(split)` with **fixed, leakage-free splits**. Use them. Each chit also has a baseline notebook in `capstone/baselines/`.

> **The budgets are targets, not walls.** They were set from one baseline run each, not tested exhaustively. If your best design misses a budget, report by how much and why,
> and what you would try next. That is graded as highly as a design that fits.

---

### Chit 1: Reading out a qubit in nanoseconds · **FPGA-ns**
A superconducting qubit is measured by sending a microwave pulse and recording what comes back. Deciding |0⟩ vs. |1⟩ must happen fast enough to feed back into
error correction. **Data:** raw ADC readout traces from a QICK/RFSoC system. **Task:** binary state discrimination. **Metric:** single-shot readout fidelity.
**Budget:** latency ≤ **50 ns**, II = 1, ≤ **16 DSPs**, LUTs ≤ **1.5 × the baseline design's synthesized LUTs** (the number is given on Monday, after a synthesis run of the baseline).
**Baseline:** fidelity 0.960. **Reference:** a published design reaches ~96% fidelity at ~32 ns. A matched filter with no ML at all (800 multiply-adds) also reaches 0.96.
Can a network beat it, or match it with fewer resources? About 6% of the |1⟩ shots decay during the readout, so no design reaches 100%.
**Lever to know about:** a shorter readout window is a faster readout. `load(window=...)` lets you cut the window (but not widen it).
**Read:** Di Guglielmo et al., *End-to-end workflow for ML-based qubit readout with QICK and hls4ml* (2025).

### Chit 2: Finding the unexpected at 40 MHz · **FPGA-L1**
Triggers keep events that *look like* known physics. An anomaly-detection trigger keeps events that look *unlike* the background, with no signal model needed.
**Data:** a simulated L1 trigger stream (background only for training) plus several new-physics signals for evaluation. **Task:** unsupervised (e.g. an autoencoder; the anomaly score is the reconstruction error).
**Metric:** signal efficiency (true-positive rate) at a false-positive rate of **10⁻⁵**, the standard for this dataset, for each signal; the headline signal is **A → 4ℓ**.
State of the art reaches about 4–5% on A → 4ℓ; **anything above 1% is a good result**. The baseline autoencoder reaches 0.11%, and a simple count of leptons, with no ML at all, reaches 0.83%: beat that.
A → 4ℓ is the headline because it stands out most, but an anomaly trigger must work for signals nobody predicted: **report and compare your designs on all four signals**, not only on A → 4ℓ.
A design that wins on A → 4ℓ and loses on the other three is not a better trigger.
**Budget:** II ≤ **5 cycles** at 200 MHz (one bunch crossing), latency ≤ **50 cycles** (250 ns), ≤ **1,000 DSPs**, LUTs ≤ **1.5 × the baseline design's synthesized LUTs** at reuse factor 5 (number given on Monday).
**Watch:** at 10⁻⁵, only 10–20 background events set the threshold, so small differences are noise: compare designs with several seeds.
**Read:** Govorkova et al., *LHC physics dataset for unsupervised New Physics detection at 40 MHz*, Sci. Data (2022).

### Chit 3: Cleaning the sky in real time · **GPU-RT** 🌍
Radio telescopes must flag radio-frequency interference (FM radio, TV, aircraft, satellites, lightning: all far brighter than the cosmic signal) before any science can be done.
**Data:** simulated observations of HERA, a radio telescope in South Africa's Karoo: each sample is one "waterfall": the signal of one dish, or of one pair of dishes, as an image of time × frequency,
with an exact mask of the interference. **Task:** per-pixel interference segmentation. **Metric:** AUPRC per pixel (F1 too).
**Why real time:** HERA flags its data after each night of observing. The next generation of arrays (SKA-Low, under construction: 512 stations, ~131,000 pairs) produces far too much data to store,
so interference must be flagged as the data arrive, before they are averaged and the raw data discarded. With time steps of about 1 s, one 512-step waterfall covers ~8 minutes;
if one GPU must keep up with all ~131,000 pairs, that is ~3.5 ms per waterfall, and the GPU has other work to do as well (calibration, averaging), so the budget is 2 ms.
**Budget:** **batch 1** on Colab's T4, p99 latency ≤ **2 ms** per waterfall *including* the copies to and from the GPU, AUPRC ≥ **0.98**, GPU memory ≤ **1 GB**. No batching.
**Baseline:** a small U-Net reaches AUPRC 0.99 at 4.5 ms p99, so it is accurate but too slow. A classical threshold flagger (provided) reaches AUPRC 0.45.
**Watch:** the waterfalls come in two groups whose brightness differs by 100–500×, so a design must work at both levels (see the baseline notebook).
**Read:** Kerrigan et al., *Optimizing sparse RFI prediction using deep learning* (2019); Mesarcik et al., *Learning to detect RFI without seeing it* (2022).

### Chit 4: A camera trap that only wakes up for animals · **MCU** 🌍
Camera traps in Karoo National Park take tens of thousands of photos, and most are empty (wind, grass, sun). An on-device filter that discards empties saves battery, storage and the
volunteers' time. **Data:** Snapshot Karoo images (downscaled to 96 × 96). **Task:** animal vs. empty. **Metric:** animal recall when at most **5% of empty images** are kept.
**Budget:** int8 weights ≤ **64 KB**, peak activations ≤ **64 KB**, ≤ **100 ms** per image at **50 M MAC/s**.
**Baseline:** recall 0.87 at 5% of empties kept (grayscale input). Colour and resolution are your choice: `load()` can return RGB, grayscale and smaller sizes.
**Watch:** class imbalance. The split is by *sequence* (photos taken in one burst stay together); the same camera sites appear in every split, which is realistic, because a camera stays where it is.
But some sites are almost always empty and others almost always busy, so a model can learn *which site it is looking at* instead of *whether there is an animal*, and that would fail at a new site.
Labels belong to the whole burst, so in some "animal" photos the animal cannot be seen at 96 × 96.
**Read:** Lin et al., *MCUNet* (2020); Banbury et al., *MLPerf Tiny* (2021).

### Chit 5: Listening for malaria mosquitoes · **MCU**
Mosquito species have characteristic wingbeat sounds. A cheap battery-powered sensor next to a bed net can listen all night, detect mosquitoes on the device, and send only a tiny summary
each morning (how many, when, ideally which species). Health teams use such counts to decide where to spray or hand out bed nets, to check whether an intervention worked, and as an early warning
when numbers rise after the rains. It has to run on the device: a night is ~12 hours of audio, of which only ~1–2% contains a mosquito, far too much to upload over a weak mobile connection.
**Data:** HumBugDB: mosquito and background recordings from lab cultures and field sites (incl. Tanzania). One sample is a **1.92-s window** of raw audio at 8 kHz.
**Task:** mosquito vs. no mosquito. **Metric:** mosquito recall at a **false-alarm rate ≤ 1%** of background windows (PR-AUC as a second number). In the field almost every window is background,
so even a few % of false alarms would swamp the real counts.
**Budget:** int8 weights ≤ **16 KB**, peak activations ≤ **32 KB**, ≤ **100 ms** per window at **50 M MAC/s** *including the feature extraction* (count it with `logmel_macs()` in `extra_macs`),
energy ≤ **4 µJ** per window (energy table in `targets.md`).
**Baseline:** recall 0.26 at 1% false alarms (PR-AUC 0.80), so there is a lot of room.
**Watch:** the test set is two whole experiments never seen in training (A: field recordings in Tanzania; B: lab cultures in the UK); the notebook prints both. The validation set shares recording
sites with training, so it is much too optimistic (0.84 recall vs. 0.26 on test): judge designs on test, with several seeds. Some sites are almost all mosquito or almost all background, so a model can
learn "which recording setup" instead of "which sound". A concrete case: in the training data, mosquito windows have most of their energy below 100 Hz (a rumble from the
recording setup), but in the test data they do not. A model that learns "low rumble = mosquito" fails on the test set. **Stretch goal:** *Anopheles* (the malaria carrier) vs. other species, testable on Test B only (`load(..., labels="species")`).
**Read:** Kiskin et al., *HumBugDB: A Large-scale Acoustic Mosquito Dataset*, NeurIPS Datasets and Benchmarks (2021).

### Chit 6: A heartbeat classifier for a wearable · **MCU**
A patch that flags abnormal beats has to last days on a coin cell. **Data:** MIT-BIH Arrhythmia Database, beat-centred windows. **Task:** 5-class beat classification (AAMI groups).
**Metric:** macro-F1 on **unseen patients**. **Budget:** int8 weights ≤ **16 KB**, peak activations ≤ **8 KB**, ≤ **5 ms** per beat at **100 M MAC/s**, energy ≤ **2 µJ** per beat (energy table in `targets.md`).
**Baseline:** macro-F1 0.47; it almost never finds the rare S and F classes, which is where the room is. Each rare class comes mostly from one or two patients,
which is a big part of why unseen patients are hard. With weights read from flash at 160 pJ per byte, energy limits you before flash size does.
**Read:** Kachuee et al., *ECG Heartbeat Classification: A Deep Transferable Representation* (2018); Moody & Mark, *The impact of the MIT-BIH Arrhythmia Database* (2001).

---
*Data licences and attributions are in each loader's docstring. Please cite the datasets in your talk.*
