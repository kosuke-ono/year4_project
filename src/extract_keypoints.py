"""Extract keypoints for the main person in each video (skill 2).

Usage: python src/extract_keypoints.py [video_id ...]    (no ids = every video in the manifest)

For each video:
  1. Run YOLO pose with tracking, so each detected person gets a track ID.
  2. Choose the main person from the first N_PICK frames: the track ID with the
     largest (box area x how central the box is), summed over those frames.
  3. Keep only that ID's keypoints. Frames where that ID is not tracked stay NaN.
     We never silently switch to another person.
Output: outputs/keypoints/<video_id>.npz (skipped if it already exists)
        outputs/keypoint_qc.csv (one row per video processed in this run)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

N_PICK = 30        # frames used to choose the main person
CONF_MIN = 0.3     # mean-confidence threshold used only for the QC "low_conf" flag

out_dir = Path("outputs/keypoints")
out_dir.mkdir(parents=True, exist_ok=True)
manifest = pd.read_csv("outputs/squat_manifest.csv")
if len(sys.argv) > 1:
    manifest = manifest[manifest.video_id.isin(sys.argv[1:])]

model = YOLO("yolo26n-pose.pt")
qc_rows = []

for row in manifest.itertuples():
    out_file = out_dir / f"{row.video_id}.npz"
    if out_file.exists():
        continue

    # 1. Track every person in every frame: frames[t] = {track_id: (area, cx, xy, conf)}
    frames, width = [], None
    for r in model.track(row.path, stream=True, persist=True, verbose=False):
        width = r.orig_shape[1]
        dets = {}
        if r.boxes.id is not None:
            ids = r.boxes.id.int().tolist()
            xyxy = r.boxes.xyxy.cpu().numpy()
            xy = r.keypoints.xy.cpu().numpy()
            conf = r.keypoints.conf.cpu().numpy()
            for i, tid in enumerate(ids):
                x1, y1, x2, y2 = xyxy[i]
                dets[tid] = ((x2 - x1) * (y2 - y1), (x1 + x2) / 2, xy[i], conf[i])
        frames.append(dets)

    # 2. Choose the main person: biggest, most central box over the first N_PICK frames
    score = defaultdict(float)
    for dets in frames[:N_PICK]:
        for tid, (area, cx, _, _) in dets.items():
            score[tid] += area * (1 - abs(cx - width / 2) / (width / 2))
    if not score:
        print(f"{row.video_id}: no tracked person in first {N_PICK} frames, skipped")
        continue
    main_id = max(score, key=score.get)

    # 3. Keep only the main person; NaN where they are not tracked
    T = len(frames)
    kpt_xy = np.full((T, 17, 2), np.nan)
    kpt_conf = np.full((T, 17), np.nan)
    for t, dets in enumerate(frames):
        if main_id in dets:
            kpt_xy[t], kpt_conf[t] = dets[main_id][2], dets[main_id][3]
            # undetected keypoints come back as (0, 0): make them NaN too
            kpt_xy[t][(kpt_xy[t] == 0).all(axis=1)] = np.nan

    np.savez(out_file, kpt_xy=kpt_xy, kpt_conf=kpt_conf, main_id=main_id)

    seen = ~np.isnan(kpt_conf[:, 0])
    other_only = sum(1 for t, d in enumerate(frames) if d and not seen[t])
    mean_conf = np.nanmean(kpt_conf[:, 11:17])  # hips, knees, ankles
    qc_rows.append(dict(video_id=row.video_id, n_frames=T, main_id=main_id,
                        n_track_ids=len({tid for d in frames for tid in d}),
                        main_detected_pct=round(100 * seen.mean(), 1),
                        frames_main_lost_others_seen=other_only,
                        mean_conf_hip_knee_ankle=round(float(mean_conf), 2),
                        low_conf=bool(mean_conf < CONF_MIN)))
    print(qc_rows[-1])

    # save after every video so an interrupted run keeps its QC rows
    qc_file = Path("outputs/keypoint_qc.csv")
    qc = pd.DataFrame(qc_rows)
    if qc_file.exists():
        old = pd.read_csv(qc_file)
        qc = pd.concat([old[~old.video_id.isin(qc.video_id)], qc])
    qc.to_csv(qc_file, index=False)
    qc_rows = []
