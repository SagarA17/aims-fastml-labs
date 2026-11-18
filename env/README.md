# Environment and data (do this BEFORE Day 2)
```
conda env create -f environment.yml
conda activate fastml
python check_env.py
```
`check_env.py` checks the packages **and downloads the jet dataset** (one time, needs internet, a few minutes).
The dataset is **required**. Every lab and assignment uses it. If it prints `All good.`, you're set.

No internet at home? Copy `jets.npz` from the course USB stick and point to it:
`export FASTML_DATA=/path/to/folder/containing/jets.npz`

Day 8 (hls4ml) uses the official hls4ml tutorial's own environment. Instructions below.
It reuses the same downloaded dataset.

## Before Day 8: the hls4ml tutorial environment (needs internet, ~1–2 GB)
```
git clone https://github.com/fastmachinelearning/hls4ml-tutorial.git
cd hls4ml-tutorial && conda env create -f environment.yml
```
You also need a C++ compiler for hls4ml's emulation (in both environments). macOS: `xcode-select --install`. Windows: use WSL.
