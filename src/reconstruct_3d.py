"""Run the AutoSoccerPose 2D -> 3D reconstruction on 20-frame shooter crops
and save an "Input | Reconstruction" GIF per clip, like 3dsp_utils/image/00001.gif.

Usage: python src/reconstruct_3d.py [crops_root] [gif_dir]
  crops_root: folders with img/001..020.jpg (default data/wonder_goals_crops;
              data/3dsp_raw/3dsp/train also works, to check against the dataset)
  gif_dir:    default football/reconstruction

Follows the repo's demo.py (gen_2d_pose) and MotionAGFormer/demo/vis_sn.py
(get_pose3D_demo) step by step, with these changes: runs on CPU instead of
CUDA, uses local copies of the model weights in data/models/, and follows
one person through the clip in 2D with left/right fixes (see track_2d)
instead of picking the person nearest the crop centre in each frame.
"""
import sys
import types
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO = Path("3D-Shot-Posture-Dataset/3dsp_utils")
sys.path[:0] = [str(REPO), str(REPO / "rtmlib")]
# vis_sn.py imports HRNet code we don't use, which needs numba, yacs, etc.
# Register an empty stand-in so we can import vis_sn's drawing functions.
stub = types.ModuleType("gen_kpts_sn")
stub.gen_video_kpts_sn = None
sys.modules["MotionAGFormer.demo.lib.hrnet.gen_kpts_sn"] = stub

from MotionAGFormer.demo.lib.utils import camera_to_world, normalize_screen_coordinates  # noqa: E402
from MotionAGFormer.demo.vis_sn import show2Dpose, show3Dpose, turn_into_clips  # noqa: E402
from MotionAGFormer.model.MotionAGFormer import MotionAGFormer  # noqa: E402
from rtmlib import Body  # noqa: E402

matplotlib.use("agg")
MODELS = Path("data/models")
RTM = MODELS / "rtmpose"
body = None  # 2D pose model, loaded in main (edit_pose.py imports this file without it)

# 3D lifting: MotionAGFormer-B with the settings from get_pose3D_demo
model = MotionAGFormer(n_layers=16, dim_in=3, dim_feat=128, dim_rep=512, dim_out=3,
                       mlp_ratio=4, act_layer=nn.GELU, attn_drop=0.0, drop=0.0, drop_path=0.0,
                       use_layer_scale=True, layer_scale_init_value=1e-5, use_adaptive_fusion=True,
                       num_heads=8, qkv_bias=False, qkv_scale=None, hierarchical=False,
                       use_temporal_similarity=True, neighbour_num=2, temporal_connection_len=1,
                       use_tcn=False, graph_only=False, n_frames=243)
# weights_only=True stops the file running code; allow the numpy scalar it stores
from numpy._core.multiarray import scalar  # noqa: E402
with torch.serialization.safe_globals([(scalar, "numpy.core.multiarray.scalar"), np.dtype,
                                       type(np.dtype("float64"))]):
    ckpt = torch.load(MODELS / "motionagformer-b-h36m.pth.tr", map_location="cpu", weights_only=True)
# The checkpoint was saved from nn.DataParallel, so names start with 'module.'
model.load_state_dict({k.removeprefix("module."): v for k, v in ckpt["model"].items()})
model.eval()


MAX_MOVE = 15  # px per frame (crop is 100 px); a bigger jump = a different person
# COCO left/right joint groups: shoulder, elbow, wrist / hip, knee, ankle
LR_GROUPS = [([5, 7, 9], [6, 8, 10]), ([11, 13, 15], [12, 14, 16])]


def body_centre(k):
    return ((k[11] + k[12]) / 2 + (k[5] + k[6]) / 2) / 2


def track_2d(imgs):
    """COCO keypoints (n_frames, 17, 2) for one person followed through the clip.

    demo.py picks, in every frame on its own, the person nearest the crop
    centre (50, 50). When the shooter is missed in a frame, that picks
    another player (0001, 0011 frame 20). Instead:
      1. anchor on the frame where someone is closest to the centre;
      2. step out from there, taking the person nearest the previous body
         centre, but only if within MAX_MOVE px; otherwise mark the frame lost;
      3. if an arm or leg pair matches the previous frame at least twice as
         well with left and right swapped, swap them back (RTMPose sometimes
         flips left/right, especially for players facing away);
      4. fill lost frames by linear interpolation between their neighbours.
    Returns keypoints, lost frame indices, swapped frame indices."""
    people = [body(img)[0] for img in imgs]  # each (n_people, 17, 2)
    dist0 = [min((np.linalg.norm(body_centre(k) - [50, 50]) for k in p), default=np.inf)
             for p in people]
    if np.isinf(min(dist0)):
        return None, list(range(len(imgs))), []
    anchor = int(np.argmin(dist0))
    kp = np.full((len(imgs), 17, 2), np.nan)
    kp[anchor] = min(people[anchor], key=lambda k: np.linalg.norm(body_centre(k) - [50, 50]))
    lost, swapped = [], []
    for direction in (1, -1):
        prev = kp[anchor].copy()
        t = anchor + direction
        while 0 <= t < len(imgs):
            moves = [np.linalg.norm(body_centre(k) - body_centre(prev)) for k in people[t]]
            if moves and min(moves) < MAX_MOVE:
                k = people[t][int(np.argmin(moves))].copy()
                for a, b in LR_GROUPS:
                    same = np.linalg.norm(k[a + b] - prev[a + b], axis=1).mean()
                    flip = np.linalg.norm(k[b + a] - prev[a + b], axis=1).mean()
                    if same > 3 and flip < 0.5 * same:
                        k[a + b] = k[b + a]
                        swapped.append(t)
                kp[t] = prev = k
            else:
                lost.append(t)
            t += direction
    ok = ~np.isnan(kp[:, 0, 0])
    for j in range(17):
        for c in range(2):
            kp[~ok, j, c] = np.interp(np.where(~ok)[0], np.where(ok)[0], kp[ok, j, c])
    return kp, sorted(lost), sorted(set(swapped))


def to_h36m(k):
    """COCO (17, 2) -> H36M-order (17, 3) with confidence 1, as demo.py does."""
    k = np.concatenate([k, np.ones((17, 1))], axis=1)  # demo.py sets confidence to 1
    hips, shoulders = (k[11] + k[12]) / 2, (k[5] + k[6]) / 2
    body_c, neck = (hips + shoulders) / 2, (k[0] + shoulders) / 2
    # Reorder COCO -> H36M exactly as demo.py does
    return np.array([hips, k[11], k[13], k[15], k[12], k[14], k[16], body_c, shoulders,
                     neck, k[0], k[6], k[8], k[10], k[5], k[7], k[9]])


def pose_3d(kp2d, w, h):
    """(20, 17, 3) 3D joints, post-processed as in get_pose3D_demo."""
    clips, downsample = turn_into_clips(kp2d[None])  # resamples 20 frames up to 243
    x = torch.from_numpy(normalize_screen_coordinates(clips[0], w=w, h=h).astype("float32"))
    with torch.no_grad():
        # The repo averages model(input_2D) with model(input_2D) (its flip
        # augmentation is never actually used), so a single pass is identical
        out = model(x)[:, downsample]
    out[:, :, 0, :] = 0
    rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521,
                    0.6223280429840088], dtype="float32")
    poses = []
    for p in out[0].numpy():
        p = camera_to_world(p, R=rot, t=0)
        p[:, 2] -= p[:, 2].min()
        poses.append(p / p.max())
    return np.array(poses)


def render(img, kp2d, kp3d):
    fig = plt.figure(figsize=(15.0, 5.4))
    ax = fig.add_subplot(121)
    ax.imshow(cv2.cvtColor(show2Dpose(kp2d, img.copy()), cv2.COLOR_BGR2RGB))
    ax.set_title("Input", fontsize=12)
    ax.axis("off")
    ax = fig.add_subplot(122, projection="3d")
    show3Dpose(kp3d, ax)
    ax.set_box_aspect(None, zoom=1.4)  # the repo crops the 3D image edges instead
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_title("Reconstruction", fontsize=12)
    plt.subplots_adjust(top=0.93, bottom=0, right=1, left=0, wspace=0)
    fig.canvas.draw()
    frame = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return frame


if __name__ == "__main__":
    crops_root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/wonder_goals_crops")
    gif_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "football/reconstruction")
    gif_dir.mkdir(parents=True, exist_ok=True)
    # 2D pose: YOLOX-x person detector + RTMPose-x, exactly the models in demo.py
    body = Body(det=str(next(RTM.glob("yolox_x*/**/end2end.onnx"))), det_input_size=(640, 640),
                pose=str(next(RTM.glob("rtmpose-x*/**/end2end.onnx"))),
                pose_input_size=(288, 384), backend="onnxruntime", device="cpu")

    for clip in sorted(p for p in crops_root.iterdir() if (p / "img").is_dir()):
        gif = gif_dir / f"{clip.name}.gif"
        if gif.exists():
            print(clip.name, "already done, skipping")
            continue
        imgs = [cv2.imread(str(p)) for p in sorted((clip / "img").glob("*.jpg"))]
        coco, lost, swapped = track_2d(imgs)
        if coco is None:
            print(clip.name, "no person found in any frame, skipped")
            continue
        kp2d = np.array([to_h36m(k) for k in coco])
        h, w = imgs[0].shape[:2]
        kp3d = pose_3d(kp2d, w, h)

        np.savez(gif_dir / f"{clip.name}_keypoints.npz", keypoints_2d=kp2d, keypoints_3d=kp3d,
                 lost_frames=lost, swapped_frames=swapped)
        frames = [render(img, a, b) for img, a, b in zip(imgs, kp2d, kp3d)]
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=100, loop=0)
        fmt = lambda s: [t + 1 for t in s] or "-"  # 1-based frame numbers
        print(f"{clip.name}: interpolated frames {fmt(lost)}, left/right fixed in {fmt(swapped)}")
