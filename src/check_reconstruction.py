"""Flag likely errors in the shooter crops and 2D/3D reconstructions.

Usage: python src/check_reconstruction.py [crops_root] [recon_dir]
Writes recon_dir/checks/<clip>.jpg (all 20 crops with the 2D skeleton,
flagged frames boxed in red) and prints a table of flags per clip.

Checks (thresholds are my choices, tune them by looking at the sheets):
  crop_jump  the shooter box centre moves > 0.4 box heights in one frame,
             or its height changes by > 40%  (crop_shooter followed someone else)
  pose_jump  the 2D body centre moves > 15 px in one frame (of a 100 px crop)
             (the pose model picked a different person inside the crop)
  lr_swap    the legs (or arms) match the previous frame much better with
             left and right swapped (pose model mixed up left and right)
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

crops_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/wonder_goals_crops")
recon_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "football/reconstruction")
(recon_dir / "checks").mkdir(exist_ok=True)

# H36M joint order used by the reconstruction (see reconstruct_3d.py)
LEFT_LEG, RIGHT_LEG = [1, 2, 3], [4, 5, 6]     # hip, knee, ankle
RIGHT_ARM, LEFT_ARM = [11, 12, 13], [14, 15, 16]  # shoulder, elbow, wrist
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]


def crop_jumps(boxes):
    cx, cy = (boxes.x1 + boxes.x2) / 2, (boxes.y1 + boxes.y2) / 2
    h = boxes.y2 - boxes.y1
    move = np.hypot(cx.diff(), cy.diff()) / h.shift()
    grow = (h / h.shift() - 1).abs()
    return set(np.where((move > 0.4) | (grow > 0.4))[0])


def pose_jumps(kp):
    move = np.linalg.norm(np.diff(kp[:, 7, :2], axis=0), axis=1)  # joint 7 = body centre
    return set(np.where(move > 15)[0] + 1)


def lr_swaps(kp):
    """Frames where swapping left/right makes the limbs match the previous
    frame at least twice as well."""
    flagged = set()
    for a, b in [(LEFT_LEG, RIGHT_LEG), (LEFT_ARM, RIGHT_ARM)]:
        for t in range(1, len(kp)):
            prev, cur = kp[t - 1, :, :2], kp[t, :, :2]
            same = np.linalg.norm(cur[a + b] - prev[a + b], axis=1).mean()
            swapped = np.linalg.norm(cur[b + a] - prev[a + b], axis=1).mean()
            if same > 3 and swapped < 0.5 * same:
                flagged.add(t)
    return flagged


def sheet(imgs, kp, flags):
    tiles = []
    for t, (img, k) in enumerate(zip(imgs, kp)):
        tile = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        for i, j in BONES:
            colour = (255, 0, 0) if i in LEFT_LEG + LEFT_ARM or j in LEFT_LEG + LEFT_ARM else (0, 0, 255)
            cv2.line(tile, tuple((k[i, :2] * 2).astype(int)), tuple((k[j, :2] * 2).astype(int)), colour, 2)
        if t in flags:
            cv2.rectangle(tile, (0, 0), (199, 199), (0, 0, 255), 4)
        cv2.putText(tile, str(t + 1), (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    return np.vstack([np.hstack(tiles[:10]), np.hstack(tiles[10:])])


rows = []
for npz in sorted(recon_dir.glob("*_keypoints.npz")):
    clip = npz.name.removesuffix("_keypoints.npz")
    kp = np.load(npz)["keypoints_2d"]
    imgs = [cv2.imread(str(p)) for p in sorted((crops_root / clip / "img").glob("*.jpg"))]
    boxes_csv = crops_root / clip / "boxes.csv"
    cj = crop_jumps(pd.read_csv(boxes_csv)) if boxes_csv.exists() else set()
    pj, sw = pose_jumps(kp), lr_swaps(kp)
    cv2.imwrite(str(recon_dir / "checks" / f"{clip}.jpg"), sheet(imgs, kp, cj | pj | sw))
    fmt = lambda s: " ".join(str(t + 1) for t in sorted(s)) or "-"  # 1-based frames
    rows.append(dict(clip=clip, crop_jump=fmt(cj), pose_jump=fmt(pj), lr_swap=fmt(sw)))

print(pd.DataFrame(rows).to_string(index=False))
print(f"\nContact sheets in {recon_dir / 'checks'} (blue = left limbs, red = right, "
      "red border = flagged frame)")
