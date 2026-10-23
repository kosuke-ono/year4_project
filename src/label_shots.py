"""Play each 3DSP shot as a looping clip and record my own label for it.

Usage: python src/label_shots.py [split_root]
Keys: a label key from LABELS below, n = skip, p = previous shot,
      space = pause, [ / ] = slower / faster, q = quit.
Labels are saved to outputs/my_labels.csv after every key press, so you
can quit at any time and it resumes at the first unlabelled shot.
"""
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Edit these to your own label scheme: key -> label written to the CSV
LABELS = {"g": "good", "b": "bad", "u": "unsure"}
SHOW_ORIGINAL_LABEL = False  # keep False so on/off target doesn't bias you
SCALE = 5                    # frames are 100x100 px, so show them at 500x500
delay_ms = 120               # per frame; the real gap is 40 ms (25 fps)

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/3dsp_raw/3dsp/train")
out = Path("outputs/my_labels.csv")
manifest = pd.read_csv("outputs/3dsp_manifest.csv", dtype={"shot": str})
original = dict(zip(manifest.shot, manifest.label))
shots = sorted(p.name for p in root.iterdir() if p.is_dir())

# Load earlier labels so we can resume
if out.exists():
    done = pd.read_csv(out, dtype={"shot": str}).set_index("shot")["my_label"].to_dict()
else:
    done = {}


def save():
    rows = [dict(shot=s, my_label=done[s]) for s in shots if s in done]
    pd.DataFrame(rows).to_csv(out, index=False)


def load_frames(shot):
    paths = sorted((root / shot / "img").glob("*.jpg"))  # 001.jpg ... 020.jpg
    return [cv2.resize(cv2.imread(str(p)), None, fx=SCALE, fy=SCALE,
                       interpolation=cv2.INTER_NEAREST) for p in paths]


def draw(frame, shot, i, n_frames):
    # Add a text panel under the frame
    panel = np.full((110, frame.shape[1], 3), 30, np.uint8)
    lines = [f"shot {shot}  ({shots.index(shot) + 1}/{len(shots)})  "
             f"labelled {len(done)}/{len(shots)}",
             f"frame {i + 1}/{n_frames}  current: {done.get(shot, '-')}",
             "  ".join(f"{k}={v}" for k, v in LABELS.items()) + "  n p q"]
    if SHOW_ORIGINAL_LABEL:
        lines.append(f"dataset: {original.get(shot)}")
    for j, text in enumerate(lines):
        cv2.putText(panel, text, (8, 25 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([frame, panel])


# Start at the first unlabelled shot
idx = next((k for k, s in enumerate(shots) if s not in done), 0)
paused = False
while 0 <= idx < len(shots):
    shot = shots[idx]
    frames = load_frames(shot)
    i, action = 0, None
    while action is None:
        cv2.imshow("label shots", draw(frames[i], shot, i, len(frames)))
        key = cv2.waitKey(delay_ms) & 0xFF
        if not paused:
            i = (i + 1) % len(frames)  # loop the clip until a key is pressed
        if key == 255:  # no key pressed
            continue
        k = chr(key)
        if k in LABELS:
            done[shot] = LABELS[k]
            save()
            action = +1
        elif k == "n":
            action = +1
        elif k == "p":
            action = -1
        elif k == " ":
            paused = not paused
        elif k == "[":
            delay_ms = min(delay_ms + 40, 1000)
        elif k == "]":
            delay_ms = max(delay_ms - 40, 20)
        elif k == "q":
            action = "quit"
    if action == "quit":
        break
    idx = max(idx + action, 0)

cv2.destroyAllWindows()
save()
print(f"Labelled {len(done)}/{len(shots)} shots -> {out} ({datetime.now():%H:%M})")
print(pd.Series(done).value_counts().to_string())
