# CLAUDE.md — EMATM5000 Technical Project (P060)

Read this file first in every session. Also read `skills.md` for reusable procedures.

## Project in one paragraph

Final-year MEng technical project (EMATM5000, 2026/27, School of Engineering Mathematics and Technology, University of Bristol).
**Project P060: "AI-based evaluation of movement skill."** Goal: a system that takes video of someone performing a sport/exercise movement and judges how well it is performed (good vs bad form), ideally with interpretable feedback.
The project description poses a genuinely open comparison between (1) **pose-estimation-based** features (joint angles, range of motion, tempo, symmetry; interpretable but fragile under occlusion) and (2) **video foundation model** embeddings (VideoMAE, InternVideo2, V-JEPA 2; rich but opaque and can latch onto confounds like venue, clothing, camera angle).

- Primary supervisor: Nawid Keshtmand. Co-supervisors: Jeff Clark, Jiayu Shang.
- Student is **not available Wednesdays or Friday afternoons**.
- Weekly 30-minute supervision meeting. Student submits an online **progress form after every meeting** (mandatory; missing forms can fail the unit).
- Supervisor said: pick **one movement in sport**, use **publicly available data** (not self-collected), and build a **simple model first**.

## Current decisions (update as they change)

- **Movement:** bodyweight **squat** (provisional; matches Paper 4's main case study; present in all public datasets considered).
- **Pose model:** Ultralytics `yolo26n-pose.pt` (17 COCO keypoints). Note: BlazePose (used in Paper 4) gives 33 landmarks including heels/feet, so ankle/foot detail from YOLO is coarser. Worth noting in the report.
- **Dataset (first):** Mendeley multi-view fitness dataset, DOI 10.17632/kgbb3yn47p.2 (v2, published 13 Jul 2026). 7 exercises, 26 subjects, front/side/diagonal views, good/bad labels, raw MP4, smartphone gym recordings. Filenames encode subject_id, exercise, form_quality, view. Using the **"Dataset Exercise Quality-wise"** folder (videos are identical across the 3 layouts). **Licence: not yet confirmed — check the bottom of the Mendeley page and record it here:** `TODO`.
- **Dataset (later, harder):** Fitness-AQA (back squat, barbell row, overhead press; real gym videos; non-commercial; check with supervisor that it is covered by the ethics allowlist first).
- **Order of attack:** side view first, then front view. Target **shallow squat (depth)** first: a static error measurable from hip/knee positions in side view. Knees-inward is much harder and needs front view.

## Key background (verified from sources)

- **Paper 4 (starting point):** Turner, Clark & Keshtmand (2026), "Markerless Pose Estimation for Resistance Training Technique Assessment", arXiv:2608.24384. BlazePose landmarks → joint-angle trajectories → RMSE vs a defined reference rep; squat is the primary case study. Performance depended strongly on camera orientation and occlusion (non-sagittal views distort 2D angles). Only the abstract has been read so far; read the full paper.
- **Fitness-AQA (arXiv:2202.14019):** squat errors = knees inward (knees buckling in), knees forward (knees too far past toes), shallow squat (glutes above knee line). Off-the-shelf pose estimators struggled on gym footage. Reported F1: shallow squat ≈ 0.83–0.87; knees inward best ≈ 0.52–0.53.
- **Sapiens (ECCV 2024):** strong open human-vision foundation model (2D pose etc.); possible alternative pose estimator later.
- COCO keypoint indices (YOLO26-pose): 0 nose, 1 L eye, 2 R eye, 3 L ear, 4 R ear, 5 L shoulder, 6 R shoulder, 7 L elbow, 8 R elbow, 9 L wrist, 10 R wrist, 11 L hip, 12 R hip, 13 L knee, 14 R knee, 15 L ankle, 16 R ankle.

## Rules for Claude Code in this repo

**The unit's AI policy ("Category 3 – selective") governs everything here.**
- AI MAY be used for: brainstorming, fact-checking, grammar checking, generating code from the student's own ideas, validating that code works.
- AI MAY NOT be used to achieve the unit's learning outcomes on the student's behalf (e.g. choosing the research design wholesale, writing report content, doing the analysis reasoning for them).
- The final report must contain an **AI-use declaration**. So **log every substantive AI contribution in `AI_USE_LOG.md`** (date, what was asked, what was produced, how the student verified or modified it). Offer to append an entry after any significant code generation.

**Working style**
1. The student wants to **learn and build this themselves**. Explain what each piece of code does, in small steps. Prefer short scripts they can read end-to-end over large frameworks.
2. **Ask before** large changes, installing heavy new dependencies, or restructuring the repo.
3. **Do not write report/thesis prose** (interim report, final report, poster). Give feedback on the student's own writing only if asked.
4. **Never invent results, numbers, citations or dataset facts.** If unsure, say so and say how to check. Distinguish verified facts from assumptions.
5. Be honest about limitations (small data, class imbalance, 2D pose from a single camera, label noise).

**Data and ethics**
- Use **public datasets only**. Never commit raw videos or datasets. `data/` and `runs/` stay in `.gitignore`.
- Respect each dataset's licence; record licence + DOI/citation in this file when known.
- Videos of real people count as personal data. Anything beyond public datasets (own recordings, new data collection) must be cleared against the Blanket Ethics Allowlist with the supervisor first.
- The final report needs an ethics statement.

**ML hygiene**
- **Split by participant (subject), never randomly by video**, to avoid leakage. Prefer `StratifiedGroupKFold` / `GroupKFold`.
- Handle class imbalance: report **F1, balanced accuracy and a confusion matrix**, not just accuracy; use `class_weight="balanced"` where appropriate.
- Always compare to a trivial majority-class baseline.
- Set random seeds; make scripts re-runnable; cache keypoints/features so pose estimation is not re-run unnecessarily.
- Filter low-confidence keypoints (undetected points can come back as (0, 0)); handle frames with no detection (NaN) explicitly.
- Evaluate side vs front views separately; Paper 4 found camera angle matters a lot.

## Environment

- macOS (zsh). Project root: `/Users/kos/Projects/year4/year4_project`.
- Python virtualenv at `.venv` (`source .venv/bin/activate`). `pip install ultralytics` done. `yolo predict model=yolo26n-pose.pt source=squat.mp4 save=True` ran successfully (CPU, ~26 ms/frame; annotated output in `runs/pose/predict`).
- Ultralytics is AGPL-3.0: fine for coursework; mention it in the report.
- HPC available if needed: BluePebble (project code `semt040104`).

## Intended repo layout

```
year4_project/
  CLAUDE.md
  skills.md
  AI_USE_LOG.md          # required: log of AI assistance
  .gitignore             # data/, runs/, .venv/, __pycache__/
  data/                  # raw datasets (not committed)
  runs/                  # YOLO outputs (not committed)
  src/                   # explore_data.py, extract_keypoints.py, features.py, train_baseline.py
  outputs/               # manifests, feature CSVs, figures, results tables
  notes/                 # supervision meeting notes + weekly summaries
```

## Plan and status

Week of Mon 12 Oct 2026 (approx. week 4 of TB1; confirm on Blackboard):
- [x] Set up venv, install Ultralytics, run `yolo26n-pose.pt` on a sample video.
- [ ] **(CURRENT)** Download Mendeley dataset (Exercise Quality-wise folder + README.md), check licence, explore file naming, build manifest (`explore_data.py`), pick subset (squat, good vs bad, side then front), count videos per subject/class, note imbalance.
- [ ] Extract keypoints for the subset → hip/knee/ankle angles per frame → per-video summary features (min knee angle, range of motion, rep duration, trunk lean) → CSV.
- [ ] First simple classifier (logistic regression / random forest) with participant-level split; report F1, balanced accuracy, confusion matrix; compare side vs front.
- [ ] One-page summary + agenda for next supervision meeting + progress form.

Later: foundation-model embeddings + linear probe for comparison; robustness (occlusion, camera angle, unseen subjects); interpretability; Fitness-AQA.

## Key deadlines (verify on Blackboard)

- Interim report (10%): Thursday of week 10.
- Draft chapter (formative): Friday week 19. Full draft report + poster/demo (formative): Friday week 21.
- Poster/demo (10%) and final report (80%, plus viva): Thursday of week 23. Poster session: Friday week 24.

## Open questions for supervisor (Nawid)

- Is the squat the right movement? Where do reliable good/bad form labels come from beyond the Mendeley set?
- Is Fitness-AQA covered by the blanket ethics allowlist?
- Should the foundation-model comparison start after the pose baseline, and what is needed for the interim report?
