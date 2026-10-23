"""Check and hand-correct the 2D pose of one clip, then redo its 3D + GIF.

Usage: python src/edit_pose.py <clip> [crops_root] [recon_dir]
       e.g. python src/edit_pose.py 0006

Window: the current frame enlarged, with the skeleton (blue = left limbs,
red = right), and all 20 frames as thumbnails underneath (click one to jump).
Keys (click the window first):
  a / d   previous / next frame
  l       swap left and right LEGS in this frame
  k       swap left and right ARMS in this frame
  i       replace this frame by the average of the frames either side
  r       reset this frame to the automatic pose
  g       show/hide the previous frame's skeleton in grey (helps spot swaps)
  s       save: re-run the 3D lifting and remake the GIF for this clip
  q / Esc quit (press twice if there are unsaved edits)
Mouse: drag any joint dot to move it. The hip centre, body centre, shoulder
centre and neck (small dots) are worked out from the other joints, as in
reconstruct_3d.py, so they move by themselves.

The automatic pose is kept in the .npz as keypoints_2d_auto, and the edited
frame numbers as edited_frames, so you can always see what was hand-changed.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from reconstruct_3d import pose_3d, render

clip = sys.argv[1]
crops_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/wonder_goals_crops")
recon_dir = Path(sys.argv[3] if len(sys.argv) > 3 else "football/reconstruction")
npz_path = recon_dir / f"{clip}_keypoints.npz"

SCALE = 6     # 100 px crops shown at 600 px
THUMB = 60
LEFT, RIGHT = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]  # H36M joint order
DERIVED = [0, 7, 8, 9]  # hip centre, body centre, shoulder centre, neck
EDITABLE = [j for j in range(17) if j not in DERIVED]
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8), (8, 9), (9, 10),
         (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)]

imgs = [cv2.imread(str(p)) for p in sorted((crops_root / clip / "img").glob("*.jpg"))]
saved = dict(np.load(npz_path))
kp = saved["keypoints_2d"].copy()                      # (20, 17, 3): x, y, confidence
auto = saved.get("keypoints_2d_auto", kp).copy()       # the untouched automatic pose
edited = set(saved.get("edited_frames", np.array([], int)).tolist())
n = len(imgs)


def update_derived(k):
    """Recompute the joints that reconstruct_3d.to_h36m builds from others."""
    k[0] = (k[1] + k[4]) / 2    # hip centre
    k[8] = (k[11] + k[14]) / 2  # shoulder centre
    k[7] = (k[0] + k[8]) / 2    # body centre
    k[9] = (k[10] + k[8]) / 2   # neck = halfway from nose to shoulder centre


def draw_skeleton(img, k, scale, thick, ghost=False):
    for i, j in BONES:
        colour = (160, 160, 160) if ghost else (255, 0, 0) if (i in LEFT or j in LEFT) else (0, 0, 255)
        cv2.line(img, tuple((k[i, :2] * scale).astype(int)), tuple((k[j, :2] * scale).astype(int)),
                 colour, thick, cv2.LINE_AA)
    if not ghost and scale == SCALE:
        for j in range(17):
            r = 3 if j in DERIVED else 6
            cv2.circle(img, tuple((k[j, :2] * scale).astype(int)), r, (0, 255, 0), -1)


def view(t, ghost, msg):
    main = cv2.resize(imgs[t], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
    if ghost and t > 0:
        draw_skeleton(main, kp[t - 1], SCALE, 2, ghost=True)
    draw_skeleton(main, kp[t], SCALE, 3)
    thumbs = []
    for s in range(n):
        th = cv2.resize(imgs[s], (THUMB, THUMB))
        draw_skeleton(th, kp[s], THUMB / 100, 1)
        if s in edited:
            cv2.rectangle(th, (0, 0), (THUMB - 1, THUMB - 1), (0, 200, 255), 2)
        if s == t:
            cv2.rectangle(th, (0, 0), (THUMB - 1, THUMB - 1), (255, 255, 255), 2)
        thumbs.append(th)
    strip = np.vstack([np.hstack(thumbs[:10]), np.hstack(thumbs[10:])])
    panel = np.full((70, main.shape[1], 3), 30, np.uint8)
    lines = [f"{clip}  frame {t + 1}/{n}   edited: {sorted(e + 1 for e in edited) or '-'}",
             "a/d frame  l legs  k arms  i interp  r reset  g ghost  s save  q quit",
             msg]
    for j, text in enumerate(lines):
        cv2.putText(panel, text, (8, 20 + 22 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([main, strip, panel])


def save():
    h, w = imgs[0].shape[:2]
    kp3d = pose_3d(kp, w, h)
    saved.update(keypoints_2d=kp, keypoints_3d=kp3d, keypoints_2d_auto=auto,
                 edited_frames=np.array(sorted(edited), int))
    np.savez(npz_path, **saved)
    frames = [render(img, a, b) for img, a, b in zip(imgs, kp, kp3d)]
    frames[0].save(recon_dir / f"{clip}.gif", save_all=True, append_images=frames[1:],
                   duration=100, loop=0)


state = dict(t=0, drag=None)


def on_mouse(event, x, y, *_):
    t = state["t"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if y < 100 * SCALE:  # main view: grab the nearest editable joint
            d = [np.hypot(*(kp[t, j, :2] * SCALE - [x, y])) for j in EDITABLE]
            if min(d) < 15:
                state["drag"] = EDITABLE[int(np.argmin(d))]
        elif y < 100 * SCALE + 2 * THUMB:  # thumbnail strip: jump to that frame
            state["t"] = min((y - 100 * SCALE) // THUMB * 10 + x // THUMB, n - 1)
    elif event == cv2.EVENT_MOUSEMOVE and state["drag"] is not None:
        kp[t, state["drag"], :2] = np.clip([x / SCALE, y / SCALE], 0, 100)
        update_derived(kp[t])
        edited.add(t)
        state["dirty"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        state["drag"] = None


cv2.namedWindow("edit pose")
cv2.setMouseCallback("edit pose", on_mouse)
ghost, msg, unsaved, quit_armed = True, "", False, False
while True:
    t = state["t"]
    cv2.imshow("edit pose", view(t, ghost, msg))
    key = cv2.waitKey(30) & 0xFF
    unsaved = unsaved or state.pop("dirty", False)  # a joint was dragged
    if key == 255:
        continue
    k = chr(key)
    msg = ""
    if k == "d":
        state["t"] = min(t + 1, n - 1)
    elif k == "a":
        state["t"] = max(t - 1, 0)
    elif k in "lk":
        a, b = ([1, 2, 3], [4, 5, 6]) if k == "l" else ([14, 15, 16], [11, 12, 13])
        kp[t, a + b] = kp[t, b + a]
        edited.add(t)
        unsaved, msg = True, f"swapped {'legs' if k == 'l' else 'arms'} in frame {t + 1}"
    elif k == "i":
        nb = [s for s in (t - 1, t + 1) if 0 <= s < n]
        kp[t] = np.mean([kp[s] for s in nb], axis=0)
        edited.add(t)
        unsaved, msg = True, f"frame {t + 1} = average of frames {[s + 1 for s in nb]}"
    elif k == "r":
        kp[t] = auto[t]
        edited.discard(t)
        unsaved, msg = True, f"frame {t + 1} reset to automatic"
    elif k == "g":
        ghost = not ghost
    elif k == "s":
        cv2.imshow("edit pose", view(t, ghost, "saving: 3D lifting + GIF, a few seconds..."))
        cv2.waitKey(1)
        save()
        unsaved, msg = False, f"saved {npz_path.name} and {clip}.gif"
        print(msg)
    elif key in (ord("q"), 27):
        if unsaved and not quit_armed:
            quit_armed, msg = True, "unsaved edits! press q again to quit without saving"
            continue
        break
    quit_armed = False

cv2.destroyAllWindows()
