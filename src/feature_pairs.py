"""Per-feature paired comparison: does each feature differ between a subject's good and bad video?

Usage: python src/feature_pairs.py
Each subject has one good and one bad video per view, so we compare them as pairs.
Prints a table per view and saves outputs/figures/pairs_<view>.png.
Test: Wilcoxon signed-rank on (bad - good) differences. With 8 features per view,
a p-value needs to be below about 0.05 / 8 = 0.006 to survive a Bonferroni correction.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

FEATURES = ["min_knee_angle", "rom", "n_reps", "rep_duration_s",
            "trunk_lean_bottom", "trunk_lean_mean", "head_trunk_bottom", "head_trunk_mean"]
df = pd.read_csv("outputs/features.csv")

for view in ["side", "front", "diagonal"]:
    d = df[df.view == view]
    good = d[d.quality == "good"].set_index("subject")[FEATURES]
    bad = d[d.quality == "bad"].set_index("subject")[FEATURES]

    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    for ax, f in zip(axes.flat, FEATURES):
        pair = pd.concat([good[f], bad[f]], axis=1, keys=["good", "bad"]).dropna()
        diff = pair.bad - pair.good
        p = wilcoxon(diff).pvalue if (diff != 0).any() else np.nan
        rows.append(dict(feature=f, n_pairs=len(pair), median_good=pair.good.median(),
                         median_bad=pair.bad.median(), median_diff=diff.median(),
                         bad_higher=f"{(diff > 0).sum()}/{len(diff)}", p_value=p))
        for _, r in pair.iterrows():
            ax.plot([0, 1], [r.good, r.bad], color="grey", alpha=0.4, marker="o", ms=3)
        ax.set_xticks([0, 1], ["good", "bad"]); ax.set_title(f, fontsize=10)
    fig.suptitle(f"{view} view: one line per subject")
    fig.tight_layout()
    fig.savefig(f"outputs/figures/pairs_{view}.png", dpi=75)

    print(f"\n===== {view} view =====")
    print(pd.DataFrame(rows).round({"median_good": 1, "median_bad": 1, "median_diff": 1,
                                    "p_value": 3}).to_string(index=False))
