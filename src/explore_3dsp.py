"""Build a manifest of the 3D Shot Posture (3DSP) train shots and check labels.

Usage: python src/explore_3dsp.py [split_root]
Each shot folder looks like: 3dsp/train/00001/{img, posture, gt, info.ini}
"""
import configparser
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/3dsp_raw/3dsp/train")
shot_dirs = sorted(p for p in root.iterdir() if p.is_dir())
print(f"Shot folders in {root}: {len(shot_dirs)}")

# 1. Read info.ini for every shot; record problems, never guess silently
rows, problems = [], []
for d in shot_dirs:
    ini = d / "info.ini"
    if not ini.exists():
        problems.append((d.name, "no info.ini"))
        continue
    cp = configparser.ConfigParser(interpolation=None)  # values may contain '%'
    cp.read(ini)
    info = dict(cp["info"]) if cp.has_section("info") else {}
    if not info.get("label"):
        problems.append((d.name, "missing/empty label"))
    rows.append(dict(shot=d.name, **info,
                     n_img=len(list((d / "img").glob("*.jpg"))),
                     n_posture=len(list((d / "posture").glob("*.json")))))
print(f"Read {len(rows)} info.ini files, problems: {len(problems)}")
for p in problems:
    print("  PROBLEM:", p)

df = pd.DataFrame(rows)

# 2. Sanity checks: folder name matches id, 20 images + 20 posture files each
print("Folder name != id field:", (df.shot != df["id"]).sum())
print("Shots without 20 img / 20 posture files:",
      ((df.n_img != 20) | (df.n_posture != 20)).sum())

# 3. Label counts (the class balance)
print("\nLabel counts:\n", df.label.value_counts(dropna=False).to_string())
print("\nLabel proportions:\n", df.label.value_counts(normalize=True).round(3).to_string())

# 4. Group by match (the 'game' field); this is the grouping variable for splits
per_game = pd.crosstab(df.game, df.label, margins=True, margins_name="total")
per_game = per_game.sort_values("total", ascending=False)
print(f"\nMatches: {df.game.nunique()}")
print("Shots per match (summary):\n", df.game.value_counts().describe().round(2).to_string())
print("\nMatch x label:\n", per_game.to_string())
one_class = per_game.drop("total").drop(columns="total")
one_class = one_class[(one_class > 0).sum(axis=1) < 2]
print(f"\nMatches with only one label: {len(one_class)} of {df.game.nunique()}")

# 5. Other fields worth knowing about before modelling
for col in ["team", "visibility", "half"]:
    print(f"\n{col} x label:\n", pd.crosstab(df[col], df.label).to_string())

# 6. Save manifest
out = Path("outputs/3dsp_manifest.csv")
out.parent.mkdir(exist_ok=True)
df.to_csv(out, index=False)
print("\nSaved", out)
