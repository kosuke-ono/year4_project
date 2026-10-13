"""First simple classifier: good vs bad, split by subject, one view at a time (skill 4).

Usage: python src/train_baseline.py
Reads outputs/features.csv. Prints accuracy, balanced accuracy, F1 (mean +- std over
folds) and a confusion matrix built from all out-of-fold predictions.
No tuning is done, so there is no tuning on the test folds.
"""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
# nested feature sets, so we can see what trunk and head posture add on top of the knee/tempo features
KNEE = ["min_knee_angle", "rom", "n_reps", "rep_duration_s"]
TRUNK = ["trunk_lean_bottom", "trunk_lean_mean"]
HEAD = ["head_trunk_bottom", "head_trunk_mean"]
FEATURE_SETS = {"knee+tempo": KNEE, "+ trunk": KNEE + TRUNK, "+ trunk + head": KNEE + TRUNK + HEAD}

df = pd.read_csv("outputs/features.csv")
df["y"] = (df.quality == "bad").astype(int)   # 1 = bad is the "positive" class

models = {
    "majority baseline": DummyClassifier(strategy="most_frequent"),
    # imputer fills NaNs with the training-fold median; scaler matters for logistic regression
    "logistic regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=1000)),
    "random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                          random_state=SEED)),
}

for view in ["side", "front", "diagonal"]:
    d = df[df.view == view].reset_index(drop=True)
    y, groups = d.y, d.subject
    print(f"\n===== {view} view: {len(d)} videos, {groups.nunique()} subjects, "
          f"{y.sum()} bad / {(1 - y).sum()} good =====")
    for (set_name, cols), (name, model) in [(fs, m) for fs in FEATURE_SETS.items() for m in models.items()]:
        X = d[cols]
        scores, y_true, y_pred = [], [], []
        for train, test in GroupKFold(n_splits=5).split(X, y, groups):
            assert not set(groups[train]) & set(groups[test])   # no subject in both
            model.fit(X.iloc[train], y.iloc[train])
            p = model.predict(X.iloc[test])
            scores.append([accuracy_score(y.iloc[test], p),
                           balanced_accuracy_score(y.iloc[test], p),
                           f1_score(y.iloc[test], p, zero_division=0)])
            y_true += list(y.iloc[test]); y_pred += list(p)
        s = np.array(scores)
        print(f"\n[{set_name}] {name}:  accuracy {s[:, 0].mean():.2f} +- {s[:, 0].std():.2f} | "
              f"balanced acc {s[:, 1].mean():.2f} +- {s[:, 1].std():.2f} | "
              f"F1(bad) {s[:, 2].mean():.2f} +- {s[:, 2].std():.2f}")
        if name != "majority baseline" and set_name == "+ trunk + head":
            cm = confusion_matrix(y_true, y_pred)
            print("  confusion matrix (rows = true, cols = predicted; order good, bad):")
            print(f"    good  {cm[0]}\n    bad   {cm[1]}")
