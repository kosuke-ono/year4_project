"""Keypoints -> knee-angle curve per video -> a few summary numbers (skill 3).

Usage: python src/features.py
Reads outputs/keypoints/*.npz and outputs/squat_manifest.csv.
Writes outputs/features.csv and outputs/figures/angle_curves.png.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

FPS = 15           # all videos are 15 fps (checked in video_info.csv)
CONF_MIN = 0.5     # keypoints below this confidence are treated as missing
SMOOTH = 5         # moving-average window in frames (about 0.33 s)

# COCO keypoint indices for each side: shoulder, hip, knee, ankle
LEFT, RIGHT = (5, 11, 13, 15), (6, 12, 14, 16)


def angle_at(b, a, c):
    """Angle in degrees at point b between the lines b->a and b->c (per frame)."""
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def video_curves(npz_path):
    d = np.load(npz_path)
    xy, conf = d["kpt_xy"], d["kpt_conf"]
    # use the leg (left or right) with the higher mean confidence
    side = max((LEFT, RIGHT), key=lambda s: np.nanmean(conf[:, list(s)]))
    sh, hip, knee, ankle = (xy[:, i] for i in side)

    knee_angle = angle_at(knee, hip, ankle)            # 180 = straight leg
    # trunk lean: angle of the hip->shoulder line from vertical (0 = upright)
    trunk = sh - hip
    trunk_lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))

    # head posture: angle at the shoulder between the hip and the ear (same side as the leg).
    # 180 = head directly in line with the trunk; smaller = head tilted forward or back.
    ear_idx = 3 if side == LEFT else 4
    head_trunk = angle_at(sh, hip, xy[:, ear_idx])

    # mask frames where any of the used keypoints is unreliable (the head has its own mask,
    # so a hidden ear does not remove the knee and trunk frames)
    bad = (conf[:, list(side)] < CONF_MIN).any(axis=1) | np.isnan(conf[:, list(side)]).any(axis=1)
    bad_head = bad | (conf[:, ear_idx] < CONF_MIN) | np.isnan(conf[:, ear_idx])
    knee_angle[bad], trunk_lean[bad], head_trunk[bad_head] = np.nan, np.nan, np.nan

    # light smoothing that ignores NaNs
    smooth = lambda x: pd.Series(x).rolling(SMOOTH, center=True, min_periods=1).mean().to_numpy().copy()
    knee_angle, trunk_lean, head_trunk = smooth(knee_angle), smooth(trunk_lean), smooth(head_trunk)
    knee_angle[bad], trunk_lean[bad], head_trunk[bad_head] = np.nan, np.nan, np.nan  # don't fill gaps
    return knee_angle, trunk_lean, head_trunk, bad, bad_head


def summarise(knee_angle, trunk_lean, head_trunk, bad, bad_head):
    # a rep = a dip in the knee angle at least 30 degrees deep, >= 10 frames apart
    filled = np.where(np.isnan(knee_angle), np.nanmax(knee_angle), knee_angle)
    bottoms, _ = find_peaks(-filled, prominence=30, distance=10)
    bottoms = bottoms[~np.isnan(knee_angle[bottoms])]
    return dict(
        min_knee_angle=np.nanmin(knee_angle),
        rom=np.nanmax(knee_angle) - np.nanmin(knee_angle),
        n_reps=len(bottoms),
        rep_duration_s=np.median(np.diff(bottoms)) / FPS if len(bottoms) > 1 else np.nan,
        # trunk and head are read at the same moments as the deepest knee angle (rep bottoms)
        trunk_lean_bottom=np.nanmedian(trunk_lean[bottoms]) if len(bottoms) else np.nan,
        trunk_lean_mean=np.nanmean(trunk_lean),
        head_trunk_bottom=np.nanmedian(head_trunk[bottoms]) if len(bottoms) else np.nan,
        head_trunk_mean=np.nanmean(head_trunk),
        missing_pct=100 * bad.mean(),
        head_missing_pct=100 * bad_head.mean(),
    )


manifest = pd.read_csv("outputs/squat_manifest.csv")
rows, curves = [], {}
for r in manifest.itertuples():
    knee, trunk, head, bad, bad_head = video_curves(f"outputs/keypoints/{r.video_id}.npz")
    curves[r.video_id] = knee
    rows.append(dict(video_id=r.video_id, subject=r.subject, view=r.view,
                     quality=r.quality, **summarise(knee, trunk, head, bad, bad_head)))

features = pd.DataFrame(rows).round(2)
features.to_csv("outputs/features.csv", index=False)
print("Saved outputs/features.csv", features.shape)
print(features.groupby(["view", "quality"])[
    ["min_knee_angle", "rom", "n_reps", "rep_duration_s", "trunk_lean_bottom", "trunk_lean_mean",
     "head_trunk_bottom", "head_trunk_mean", "head_missing_pct"]
].median().round(1).to_string())

# sanity-check plot: side view, subjects 1-3, good vs bad
fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharey=True)
for i, s in enumerate([1, 2, 3]):
    for j, q in enumerate(["good", "bad"]):
        k = curves[f"subject_{s:03d}_squat_{q}_side"]
        axes[i, j].plot(np.arange(len(k)) / FPS, k)
        axes[i, j].set_title(f"subject {s} {q} (side)")
        axes[i, j].set_ylabel("knee angle (deg)")
axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
fig.tight_layout()
fig.savefig("outputs/figures/angle_curves.png", dpi=80)
