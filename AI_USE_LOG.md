# AI use log

One entry per substantive AI contribution (date, prompt, what was produced, files, how I checked/changed it).


## 2026-10-22 — 3DSP dataset label check (Claude Code)
- **Asked:** load the 3D Shot Posture dataset, count on/off-target shots, confirm all 200 train shots have a label, group shots by match (`game` in info.ini).
- **Produced:** `src/explore_3dsp.py` (reads every info.ini with configparser, checks labels and file counts, match x label crosstab, saves `outputs/3dsp_manifest.csv`). Unzipped `3dsp.zip` into `data/3dsp_raw/`. Added `3D-Shot-Posture-Dataset/` to `.gitignore`.
- **Output:** 200/200 labelled; 110 on target, 90 off target; 20 matches (3–18 shots each), 3 matches with a single label.
- **How I checked / changed it:** TODO (e.g. opened a few info.ini files by hand and compared to the manifest).
