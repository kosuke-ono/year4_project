"""Cut 3DSP-style 100x100 shooter crops (20 frames at 25 fps) from my own clips.

Usage: python src/crop_shooter.py [video_dir] [out_dir]
For each video:
  1. Scrub to the frame where the foot hits the ball, then press Enter.
     Keys: d / a = +1 / -1 frame, D / A = +10 / -10, s = skip video, q = quit.
  2. Click the shooter (YOLO person boxes are drawn on the contact frame).
The script then follows that player through the other frames (nearest box
centre) and saves out_dir/<video>/img/001..020.jpg, boxes.csv and info.ini.

To match 3DSP (from its README, info.ini and dataset_annotation.py):
  - frames start 500 ms before the shot and step 40 ms (25 fps), 20 frames kept
  - each crop is a fixed 100x100 window centred on the shooter's box, black
    padding where it runs off the frame
  - my addition: frames are rescaled so the shooter's box is TARGET_HEIGHT px
    tall (median shooter box height across the 200 train shots), because
    screen recordings are a different resolution/zoom from SoccerNet video
"""
import configparser
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

N_FRAMES, STEP_MS, BEFORE_MS = 20, 40, 500
CROP = 100
TARGET_HEIGHT = 76.5  # set to None to keep the recording's own scale
VIEW_SCALE = 0.5      # recordings are 2532x1170; shrink them for the screen
ZOOM = 2              # extra detection pass on the centre 1/ZOOM of the frame

video_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "wonder_goals")
out_root = Path(sys.argv[2] if len(sys.argv) > 2 else "data/wonder_goals_crops")
model = YOLO("yolo26n-pose.pt")


def read_timestamps(path):
    """Timestamp (ms) of every frame. Screen recordings have a variable frame
    rate, so we use real timestamps rather than frame_number / fps."""
    cap = cv2.VideoCapture(str(path))
    times, previews = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        times.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        previews.append(cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE))
    cap.release()
    return np.array(times), previews


def read_frames(path, wanted):
    """Full-resolution copies of the frame indices in `wanted`."""
    cap = cv2.VideoCapture(str(path))
    frames, i = {}, 0
    while len(frames) < len(set(wanted)):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            frames[i] = frame
        i += 1
    cap.release()
    return [frames[i] for i in wanted]


def pick_contact(name, times, previews):
    i = 0
    while True:
        view = previews[i].copy()
        cv2.putText(view, f"{name}  frame {i}/{len(previews) - 1}  {times[i]:.0f} ms  "
                    "a/d A/D scrub, Enter = ball contact, s skip, q quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("crop shooter", view)
        # Short waits in a loop (not waitKey(0)) so Ctrl+C in the terminal works
        k = -1
        while k == -1:
            k = cv2.waitKey(50)
        k &= 0xFF
        if k == 27:  # Esc
            k = ord("q")
        step = {ord("d"): 1, ord("a"): -1, ord("D"): 10, ord("A"): -10}.get(k, 0)
        i = int(np.clip(i + step, 0, len(previews) - 1))
        if k in (13, 10):
            return i
        if k in (ord("s"), ord("q")):
            return chr(k)


def yolo_boxes(image):
    return model(image, imgsz=1920, conf=0.1, verbose=False)[0].boxes.xyxy.cpu().numpy()


def iou(a, b):
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / union


def detect(frame):
    """Person boxes as an (n, 4) array of x1, y1, x2, y2 in full-res pixels.
    Players are small in wide broadcast shots, so the default 640 px input
    finds almost nobody; 1920 px with a low threshold finds most players.
    A second pass on the centre of the frame, zoomed in ZOOM times, finds
    players in crowded groups that the full-frame pass misses (e.g. 0001.mov)."""
    boxes = list(yolo_boxes(frame))
    h, w = frame.shape[:2]
    x0, y0 = int(w * (1 - 1 / ZOOM) / 2), int(h * (1 - 1 / ZOOM) / 2)
    for b in yolo_boxes(frame[y0:h - y0, x0:w - x0]) + [x0, y0, x0, y0]:
        if all(iou(b, kept) < 0.5 for kept in boxes):  # skip duplicates
            boxes.append(b)
    return np.array(boxes).reshape(-1, 4)


def click_shooter(frame, boxes):
    view = cv2.resize(frame, None, fx=VIEW_SCALE, fy=VIEW_SCALE)
    for x1, y1, x2, y2 in boxes * VIEW_SCALE:
        cv2.rectangle(view, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(view, "click the shooter's green box (click elsewhere = draw it yourself, "
                "any key = skip)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    clicked = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.append((x / VIEW_SCALE, y / VIEW_SCALE))

    cv2.setMouseCallback("crop shooter", on_mouse)
    cv2.imshow("crop shooter", view)
    while not clicked:
        if cv2.waitKey(50) != -1:
            return None
    cv2.setMouseCallback("crop shooter", lambda *_: None)
    x, y = clicked[0]
    inside = [j for j, (x1, y1, x2, y2) in enumerate(boxes) if x1 <= x <= x2 and y1 <= y <= y2]
    if inside:
        return boxes[inside[0]]
    # No detection there: drag a box round the shooter, then press Enter
    x, y, w, h = cv2.selectROI("crop shooter", view, showCrosshair=False)
    if w == 0 or h == 0:
        return None
    return np.array([x, y, x + w, y + h]) / VIEW_SCALE


def centre(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def follow(all_boxes, start, start_box):
    """From the clicked box, step forwards and backwards in time, each time
    taking the detected box whose centre is nearest the previous one. A box
    only counts if its centre is within 0.4 box-heights and its height within
    40% of the previous box (looser limits let it jump to a nearby player,
    e.g. 0005). Otherwise reuse the previous box (3DSP reuses the nearest
    frame's box too)."""
    track = [None] * len(all_boxes)
    track[start] = start_box
    for direction in (1, -1):
        prev = start_box
        j = start + direction
        while 0 <= j < len(all_boxes):
            b = all_boxes[j]
            if len(b):
                h_prev = prev[3] - prev[1]
                dist = np.linalg.norm([centre(x) - centre(prev) for x in b], axis=1)
                size_ok = np.abs((b[:, 3] - b[:, 1]) / h_prev - 1) < 0.4
                dist[~size_ok] = np.inf
                if dist.min() < 0.4 * h_prev:
                    prev = b[np.argmin(dist)]
            track[j] = prev
            j += direction
    return np.array(track)


def crop(frame, box, scale):
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    frame = cv2.copyMakeBorder(frame, CROP, CROP, CROP, CROP, cv2.BORDER_CONSTANT, value=0)
    cx, cy = centre(box) * scale + CROP  # + CROP because of the border
    x0, y0 = int(cx - CROP / 2), int(cy - CROP / 2)
    return frame[y0:y0 + CROP, x0:x0 + CROP]


videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in (".mov", ".mp4"))
for path in videos:
    out = out_root / path.stem
    if (out / "info.ini").exists():
        print(path.name, "already done, skipping")
        continue
    times, previews = read_timestamps(path)
    contact = pick_contact(path.name, times, previews)
    if contact == "q":
        break
    if contact == "s":
        continue

    # 20 target times, then the real frame nearest each one
    targets = times[contact] - BEFORE_MS + STEP_MS * np.arange(N_FRAMES)
    idx = [int(np.argmin(np.abs(times - t))) for t in targets]
    outside = (targets < times[0] - STEP_MS / 2) | (targets > times[-1] + STEP_MS / 2)
    if outside.any():
        print(f"  WARNING {path.name}: {outside.sum()} target times fall outside the "
              "recording, so frames are repeated at the ends")

    frames = read_frames(path, idx)
    all_boxes = [detect(f) for f in frames]
    c = int(np.argmin(np.abs(targets - times[contact])))  # contact in the 20
    shooter = click_shooter(frames[c], all_boxes[c])
    if shooter is None:
        print(f"  {path.name}: no shooter box picked, skipped")
        continue
    track = follow(all_boxes, c, shooter)

    heights = track[:, 3] - track[:, 1]
    scale = TARGET_HEIGHT / np.median(heights) if TARGET_HEIGHT else 1.0
    (out / "img").mkdir(parents=True, exist_ok=True)
    crops = [crop(f, b, scale) for f, b in zip(frames, track)]
    for k, im in enumerate(crops, start=1):
        cv2.imwrite(str(out / "img" / f"{k:03d}.jpg"), im)
    # All 20 crops side by side, to check the shooter stays centred (camera cuts!)
    cv2.imwrite(str(out / "preview.jpg"), np.hstack(crops))

    pd.DataFrame(dict(frame=range(1, N_FRAMES + 1), source_frame=idx,
                      source_ms=times[idx].round(1), x1=track[:, 0], y1=track[:, 1],
                      x2=track[:, 2], y2=track[:, 3])).to_csv(out / "boxes.csv", index=False)
    cp = configparser.ConfigParser(interpolation=None)
    cp["info"] = dict(id=path.stem, source=str(path), label="Shots on target",
                      contact_frame=str(contact), contact_ms=f"{times[contact]:.1f}",
                      position_step=str(STEP_MS), scale=f"{scale:.4f}",
                      median_box_height=f"{np.median(heights):.1f}",
                      frames_outside_recording=str(int(outside.sum())))
    with open(out / "info.ini", "w") as fh:
        cp.write(fh)
    print(f"  {path.name}: saved 20 crops to {out} (scale {scale:.3f})")

cv2.destroyAllWindows()
