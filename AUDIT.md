# RailGuard-FSL++ — Code Audit & Fixes

Review of the backend for correctness bugs, crash/demo risks, and the weak
evaluation numbers. Items marked **FIXED** were changed in this pass; numpy-only
modules were re-tested (13/13 existing tests pass) and the vision gates were
exercised on generated frames.

## Critical (would block a judge running the repo)

### 1. Missing `requirements.txt` — **FIXED**
`README.md` told users to `pip install -r requirements.txt`, but only
`pyproject.toml` / `poetry.lock` existed. Anyone not using Poetry hit a wall on
step one. Added `requirements.txt` mirroring the pyproject dependencies.

### 2. No way to run without downloading datasets — **FIXED (new capability)**
There was no bundled or generated data, so `train.py`/the API/the dashboard
could not run until someone downloaded Mendeley/RSDDs and arranged folders.
Added `scripts/make_synthetic_data.py`, which writes class-distinct synthetic
rail frames into `data/mendeley/{healthy,crack,...}`. Verified end-to-end:
**54/54 generated frames pass phase-1 validation and phase-2 rail extraction.**
This lets the whole pipeline + UI run for a demo with zero downloads. (Use the
real datasets for reportable numbers.)

## High (wrong results / weak numbers)

### 3. Open-set rejection collapses → 7% unknown-detection — **FIXED**
`results/experiment_results.json` showed `open_set_detection_rate = 0.072`.
Root cause: `PrototypicalNetwork.open_set_classify` thresholded the
nearest-prototype distance at `calibrated_threshold() = mean + 3*std` of the
**support-to-own-prototype** distances. Those distances are a biased (too small)
estimate of where genuine *known queries* land, and on the unit sphere the rule
sits at the wrong place and **collapses to all-known (recall ≈ 0.07) or
all-unknown (recall ≈ 1.0, false-unknown ≈ 1.0) depending on feature scale** —
both reproduced with a numpy harness.

Fix: added `PrototypicalNetwork.calibrate_open_set(calib_embeddings,
false_unknown_budget)`, a distribution-free (conformal-style) threshold set to
the `(1 - budget)` quantile of **held-out known** nearest-distances.
`evaluate_few_shot(..., open_set_budget=0.1, n_calib_shots=1)` and `train.py`
(`--open-set-budget`, `--n-calib-shots`) now use it. The old rule is preserved
as the default (back-compatible) and reported as the ablation baseline.

Measured on a representative separable embedding space (closed-set ≈ 0.81),
5-way 5-shot, 200 episodes:

| rule | unknown recall | false-unknown |
|------|---------------:|--------------:|
| `mean + 3σ` (old default) | collapses (≈0.07 or ≈1.0) | uncontrolled |
| calibrated, budget 0.10 | **0.47** | 0.16 |
| calibrated, budget 0.20 | **0.59** | 0.24 |

Numbers on the real rail embeddings will differ, but the operating point is now
**tunable and honest** instead of an accidental magic constant. Re-run
`python train.py --mode few-shot` to regenerate `models/few_shot_eval.json` with
both the baseline and calibrated rows.

## Medium (surface for the demo)

### 4. Frame validator rejects non-horizontal rail frames — **documented**
`InputValidator.detect_rail_presence` requires **>3 near-horizontal long lines**
(`HoughLinesP`, `minLineLength=100`). Frames whose rails run vertically are
rejected as `no_rail_detected`. This is why the synthetic generator emits
*horizontal* rails. If a real demo image returns `no_rail_detected`, it is
almost always this gate — rotate the frame 90° or lower the threshold. Consider
making the check orientation-agnostic (count long lines whose angle is within
±20° of *either* axis) before relying on it with arbitrary footage.

### 5. Closed-set accuracy 0.547 is below the >0.80 target — **recommendation**
Not a bug, but worth pushing. Levers, cheapest first: (a) raise shots where data
allows; (b) embed the whole rail crop's CLS in addition to the top-K
anomalous-patch mean (the current `--mode embed` averages the 3 most-anomalous
patches, which can be noisy on classes whose signal is diffuse); (c) try the
ViT-B/14 backbone (`vitb14`) — bigger DINOv2 features usually separate these
textures better; (d) mean (not median) prototypes once shots ≥ 5.

## Verified NOT bugs
- `lucide-react@^1.18.0` resolves (1.18.0 is the current latest; lucide moved to
  1.x). `npm install` is fine.
- One-embedding-per-image eval + perceptual-hash dedup are implemented correctly
  (no patch-level support/query leakage).
- `tests/` (numpy-only: prototypical + conformal) pass after the changes.

## Files changed
- `requirements.txt` *(new)*
- `scripts/make_synthetic_data.py` *(new)*
- `backend/models/prototypical_network.py` — `calibrate_open_set`,
  `open_set_threshold`, calibrated branch in `evaluate_few_shot`
- `train.py` — `--open-set-budget`, `--n-calib-shots`, calibrated open-set row
- `SETUP_AND_RUN.md` *(new)* — desktop terminal walkthrough
