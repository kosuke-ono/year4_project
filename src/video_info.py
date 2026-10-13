"""Read basic metadata (fps, size, length, rotation) for every video in the manifest.

Usage: python src/video_info.py
Reads outputs/squat_manifest.csv, writes outputs/video_info.csv.
"""
import cv2
import pandas as pd

manifest = pd.read_csv("outputs/squat_manifest.csv")

rows = []
for r in manifest.itertuples():
    cap = cv2.VideoCapture(r.path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows.append(dict(video_id=r.video_id, view=r.view, quality=r.quality,
                     fps=fps, width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     rotation=cap.get(cv2.CAP_PROP_ORIENTATION_META),
                     n_frames=n_frames, duration_s=n_frames / fps if fps else None))
    cap.release()

info = pd.DataFrame(rows)
info.to_csv("outputs/video_info.csv", index=False)

print("fps:\n", info.fps.round(2).value_counts().to_string())
print("\nresolution (w x h) / rotation metadata:\n",
      info.groupby(["width", "height", "rotation"]).size().to_string())
print("\nduration (s) by view and quality:\n",
      info.groupby(["view", "quality"]).duration_s.describe().round(1).to_string())
