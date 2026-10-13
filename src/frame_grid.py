"""Make a frame grid with the pose skeleton drawn on, for a visual sanity check (skill 5).

Usage: python src/frame_grid.py <video_id> [n_frames]
Saves outputs/figures/grid_<video_id>.png: n_frames evenly spaced frames, with the
skeleton drawn by YOLO and the frame number in the title.
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

video_id = sys.argv[1]
n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 8

manifest = pd.read_csv("outputs/squat_manifest.csv").set_index("video_id")
cap = cv2.VideoCapture(manifest.loc[video_id, "path"])
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
model = YOLO("yolo26n-pose.pt")

fig, axes = plt.subplots(2, (n_frames + 1) // 2, figsize=(3 * ((n_frames + 1) // 2), 8))
for ax, idx in zip(axes.flat, [int(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue
    result = model(frame, verbose=False)[0]
    ax.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
    ax.set_title(f"frame {idx}  ({len(result.boxes)} person)", fontsize=9)
    ax.axis("off")
fig.suptitle(video_id)
fig.tight_layout()

out = Path("outputs/figures")
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / f"grid_{video_id}.png", dpi=80)
print("Saved", out / f"grid_{video_id}.png")
