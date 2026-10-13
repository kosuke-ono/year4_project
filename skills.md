# skills.md — Reusable procedures for the P060 project

Each skill is a short, repeatable procedure. Follow it step by step, explain as you go, and log substantive AI help in `AI_USE_LOG.md`. See `CLAUDE.md` for project context and rules.

---

## 1. Explore a dataset and build a manifest

**When:** a new dataset has been downloaded into `data/`.

1. Read the dataset README and record licence, DOI and citation in `CLAUDE.md`.
2. List file extensions and counts; show a sample of 20 paths.
3. Parse the filename convention (subject, exercise, quality, view) into a table. Print any rows that could not be parsed. Do not guess silently.
4. Save `outputs/<dataset>_manifest.csv`.
5. Print crosstabs: quality × view; subject × quality. Flag subjects with only one class and any strong imbalance.
6. Summarise in plain language what the numbers mean for evaluation.

## 2. Extract keypoints from videos

**When:** a manifest exists and the subset has been chosen.

1. Use `yolo26n-pose.pt` via the Python API (`stream=True`), one video at a time.
2. Per frame, store the 17 keypoints (x, y) and confidence for the main person (largest box or highest confidence), with NaN where nothing is detected.
3. Save per-video output to `outputs/keypoints/<video_id>.npz`. Skip videos already processed.
4. Report detection rate (% frames with a person) and mean keypoint confidence per video; flag low-quality videos.
5. Do not drop or interpolate frames without telling the student.

## 3. Compute joint angles and per-video features

**When:** keypoints are saved.

1. Choose the side with the better confidence in side view (left or right hip/knee/ankle: 11/13/15 or 12/14/16).
2. Compute knee angle (hip–knee–ankle) and, if shoulders are used, trunk lean. Mask low-confidence frames; smooth lightly (e.g. a short moving average or Savitzky–Golay) and say so.
3. Summary features per video: minimum knee angle (depth), range of motion, rep duration, plus optional hip-below-knee indicator.
4. Save `outputs/features_<view>.csv` with subject, quality, view and features.
5. Plot a few example angle-vs-time curves (good vs bad) so the student can sanity-check the features by eye.

## 4. Train and evaluate a baseline classifier

**When:** a features CSV exists.

1. Start with a majority-class baseline, then logistic regression and a random forest.
2. **Split by subject** (`GroupKFold` or `StratifiedGroupKFold`). Never split randomly by video.
3. Use `class_weight="balanced"`.
4. Report F1, balanced accuracy, confusion matrix, and the spread across folds. Give side and front results separately.
5. State limitations (sample size, label noise, 2D-only, per-subject variation) and what would change the conclusion.
6. Do not tune on the test fold.

## 5. Visual sanity check of pose output

**When:** after any new video type or pose model is used.

1. Render annotated videos or frame grids for a handful of good and bad examples.
2. Check: skeleton on hips/knees/ankles throughout? Left/right swaps in side view? Missing detections? Other people confusing the model? Occlusion?
3. Write the findings as bullet notes in `notes/` for the student to use in the report.

## 6. Prepare for a supervision meeting

**When:** a day or two before the weekly 30-minute meeting.

1. Summarise progress since last meeting (what was run, what the numbers were, what broke), from the repo and `notes/`, in half a page.
2. List 3–5 specific questions or decisions needed from the supervisor.
3. Draft an agenda the student can send 24 hours ahead. The student edits and sends it.
4. After the meeting, remind the student to submit the **progress form** and to record action points in `notes/`.

## 7. Log AI assistance

**When:** after any substantive AI-generated code or explanation.

Append to `AI_USE_LOG.md`: date, prompt in one line, what was produced, which file(s), and how the student checked or changed it. The final report needs an AI-use declaration built from this log.

## 8. Next-step experiments (only after the baseline works)

- Extract embeddings from a video foundation model (e.g. VideoMAE, V-JEPA 2) and train a linear probe with the same subject-level splits for a fair comparison against pose features.
- Test robustness: front vs side vs diagonal, unseen subjects, occlusion. Check whether embeddings track the view or background instead of form.
- Try a second pose estimator (Sapiens or BlazePose) and compare.
- Move to Fitness-AQA only after confirming the ethics allowlist with the supervisor.
