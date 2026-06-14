# Advanced capabilities (from the research report → into the product)

Three research-grade features from `RailGuard-FSL++_Research_Report.pdf` are now
implemented end-to-end (backend + API + UI). All three are exercised by the new
**Safety & Calibration** page (`/safety`) in the dashboard, and the math/CV cores
are covered by `tests/test_safety.py` (6 tests, no GPU needed).

## 1. Conformal recall guarantee
A distribution-free, finite-sample lower bound on detection recall — the
safety-critical metric (a missed crack costs far more than a false alarm).

- Core: `backend/utils/conformal.py` — `conformal_recall_threshold()` picks the
  operating threshold from held-out **defect** anomaly scores so that
  `P(defect detected) ≥ target_recall` holds (split conformal prediction).
- The orchestrator accumulates anomaly scores at inference
  (`_defect_scores` / `_healthy_scores_obs`) and `calibrate_conformal()` applies
  the threshold to the live detector.
- API: `POST /conformal/calibrate {target_recall}` · `GET /conformal/status`.
- UI: target-recall slider → shows **guaranteed recall, empirical recall,
  false-positive rate, calibration n**, and whether the guarantee is achievable.

## 2. Cross-domain calibration
Different railways/cameras/lighting shift the score scale; re-pick the threshold
per domain at a healthy percentile — no retraining.

- `orchestrator.calibrate_threshold(percentile, domain)` →
  `PatchCoreDetector.recalibrate()` + an operating-point readout (FPR / recall).
- API: `POST /calibrate/threshold {percentile, domain}` · `GET /calibrate/domains`.
- UI: per-domain threshold + FPR table with a percentile slider.

## 3. Synthetic defect augmentation
Expand a scarce defect class by synthesizing new examples — the scarce-positive
fix from the report.

- `backend/datasets/synthetic_augment.py`:
  - **CutPaste** (`cutpaste()`): paste a defect patch onto a healthy crop with
    seamless (Poisson) or feathered blending (Li et al., CVPR 2021).
  - **Procedural** (`synthesize_defect()`): draw crack / squat / spalling / groove.
  - `expand_support()`: grow a synthetic support set for few-shot.
  - GAN/diffusion generators (DefectGAN, DFMGAN, AnomalyDiffusion) drop in behind
    the same `synthesize_defect(...)` signature.
- API: `POST /augment/preview` (file + `kind`) → original + synthetic PNGs.
- UI: pick a defect kind, drop a healthy crop, see original vs synthetic.

## Try it in the demo
After the API is up and you've run a few detections from the Upload page:

```bash
# conformal
curl -X POST localhost:8000/conformal/calibrate -H 'Content-Type: application/json' -d '{"target_recall":0.95}'
# cross-domain
curl -X POST localhost:8000/calibrate/threshold -H 'Content-Type: application/json' -d '{"percentile":97,"domain":"RSDDs"}'
# augmentation
curl -X POST localhost:8000/augment/preview -F file=@data/mendeley/healthy/healthy_000.png -F kind=crack -o /dev/null -w "%{http_code}\n"
```

Or just open the dashboard → **Safety & Calibration**.
