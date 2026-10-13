"""Build a manifest of the squat videos and print class counts (skill 1).

Usage: python src/explore_data.py [video_root]
Filenames look like: subject_012_squat_bad_side.mp4
"""
import re
import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1] if len(sys.argv) > 1 else "squat")
pattern = re.compile(r"subject_(\d+)_(\w+?)_(good|bad)_(front|side|diagonal)\.mp4$")

# 1. File extensions and a sample of paths
all_files = [p for p in root.rglob("*") if p.is_file() and p.name != ".DS_Store"]
print("Extensions:", pd.Series([p.suffix for p in all_files]).value_counts().to_dict())
files = sorted(p for p in all_files if p.suffix == ".mp4")  # skip README etc.
print("Sample paths:")
for p in files[:20]:
    print("  ", p)

# 2. Parse filenames; never guess silently
rows, unparsed = [], []
for p in files:
    m = pattern.match(p.name)
    if m is None:
        unparsed.append(p)
        continue
    subject, exercise, quality, view = m.groups()
    rows.append(dict(video_id=p.stem, path=str(p), subject=int(subject),
                     exercise=exercise, quality=quality, view=view,
                     folder_quality=p.parent.name))
print(f"\nParsed {len(rows)} files, could not parse {len(unparsed)}")
for p in unparsed:
    print("  UNPARSED:", p)

df = pd.DataFrame(rows)
# Sanity check: the quality in the filename should match the folder it sits in
mismatch = df[df.quality != df.folder_quality]
print("Folder/filename quality mismatches:", len(mismatch))

# 3. Save manifest
out = Path("outputs/squat_manifest.csv")
out.parent.mkdir(exist_ok=True)
df.drop(columns="folder_quality").to_csv(out, index=False)
print("Saved", out)

# 4. Crosstabs
print("\nQuality x view:\n", pd.crosstab(df.quality, df.view))
print("\nSubject x quality (all views):\n", pd.crosstab(df.subject, df.quality))
for view, g in df.groupby("view"):
    ct = pd.crosstab(g.subject, g.quality)
    print(f"\nSubject x quality, {view} view:\n", ct.T)
    one_class = ct[(ct > 0).sum(axis=1) < 2]
    print("Subjects with only one class:", list(one_class.index))
print("\nSubjects:", df.subject.nunique(), "| missing 1-26:",
      sorted(set(range(1, 27)) - set(df.subject)))
