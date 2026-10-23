"""Turn every 3DSP shot into a looping GIF and an HTML page showing them all in order.

Usage: python src/make_shot_gallery.py [split_root] [ms_per_frame]
Open runs/3dsp_gallery/index.html in a browser. Each shot appears at its
original 100x100 crop size and enlarged 3x, with its shot id for pen-and-paper notes.
Output goes in runs/ (git-ignored) because the frames show real people.
"""
import sys
from pathlib import Path

from PIL import Image

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/3dsp_raw/3dsp/train")
ms = int(sys.argv[2]) if len(sys.argv) > 2 else 120  # real gap is 40 ms (25 fps)
out = Path("runs/3dsp_gallery")
(out / "gifs").mkdir(parents=True, exist_ok=True)

shots = sorted(p.name for p in root.iterdir() if p.is_dir())
for shot in shots:
    paths = sorted((root / shot / "img").glob("*.jpg"))  # 001.jpg ... 020.jpg
    frames = [Image.open(p).convert("RGB") for p in paths]
    frames[0].save(out / "gifs" / f"{shot}.gif", save_all=True,
                   append_images=frames[1:], duration=ms, loop=0)

# One card per shot: original size, then the same GIF scaled up 3x
cards = "\n".join(
    f'<figure><img src="gifs/{s}.gif" width="100">'
    f'<img src="gifs/{s}.gif" width="300" class="big">'
    f"<figcaption>{s}</figcaption></figure>" for s in shots)
html = f"""<!doctype html><meta charset="utf-8"><title>3DSP shots</title>
<style>
body {{ font-family: sans-serif; background: #222; color: #eee; margin: 16px; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 16px; }}
figure {{ margin: 0; background: #333; padding: 8px; display: flex;
          align-items: flex-end; gap: 8px; flex-wrap: wrap; width: 416px; }}
.big {{ image-rendering: pixelated; }}
figcaption {{ width: 100%; font-size: 20px; font-weight: bold; }}
</style>
<p>{len(shots)} shots from {root}, {ms} ms per frame (real speed is 40 ms).</p>
<div class="grid">
{cards}
</div>
"""
(out / "index.html").write_text(html)
print(f"Wrote {len(shots)} GIFs and {out / 'index.html'}")
