# RailGuard-FSL++

**Few-Shot Defect Detection in Rail Infrastructure**

A research-grade AI platform combining self-supervised representation learning (DINOv2), healthy-only
anomaly detection (PatchCore on dense patch tokens), few-shot classification (prototypical networks),
open-set defect discovery, conformal guaranteed-recall calibration, and a predictive-maintenance digital twin.

## Core Research Question

> How can railway defects be detected when defect examples are extremely rare and new defect categories
> continuously appear?

**Answer:** zero defect examples to *detect* (healthy-only memory bank), five to *name* (prototypes),
and a distribution-free statistical guarantee on recall (conformal calibration).

## Architecture

```
Camera/video frames → Frame validity gate (blur + rail-presence + occlusion check)
  → Rail region crop → Overlapping 256px patches → DINOv2 encoder
    → Stage 1: PatchCore on dense patch tokens (16×16 grid per patch), healthy-only memory bank
      → (normal → Pass)
      → (anomalous) → Stage 2: Prototypical classifier on CLS embeddings of anomalous patches only
        → hard-negative suppression (welds, joints, fasteners, stains)
        → (near prototype → Defect type + severity + digital twin alert)
        → (far from all → Unknown anomaly → Discovery page → label → new few-shot class)
```

Key design decisions (each maps to a failure mode of naive approaches):

| Decision | Why |
|----------|-----|
| Dense patch tokens for stage 1 (AnomalyDINO-style) | Hairline cracks <1% of frame aren't averaged away; finer heatmaps |
| k-center greedy coreset memory bank | Covers the healthy manifold; random sampling misses rare-but-healthy structures |
| Stage 2 classifies only anomalous patches | Tiny defect signal isn't diluted by healthy patches |
| Median prototypes on L2-normalized embeddings | Robust to one bad support shot; distances comparable across domains |
| Per-episode calibrated open-set threshold | "Unknown" rejection adapts to the support set instead of a magic constant |
| Conformal recall calibration | P(defect detected) ≥ target, finite-sample, distribution-free |
| One embedding per *image* (never per patch) in eval | No support/query leakage from patches of the same physical defect |
| Perceptual-hash dedup before splits | Kaggle sets contain near-duplicate frames that inflate metrics |
| Threshold + decision on the same raw 1-NN scale | Calibration is honest; smoothed maps are visualization-only |

## Setup

```bash
# Backend (Python 3.10+) — pip/venv path (recommended)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# ...or with Poetry (regenerates the lock on first install)
poetry install

# Frontend (separate terminal)
cd frontend
npm install     # or pnpm install
```

> Note: if Poetry errors with a dyld/Python-framework message, your Poetry installation references a
> removed Homebrew Python — reinstall it (`curl -sSL https://install.python-poetry.org | python3 -`)
> or just use the pip/venv path above.

The first run downloads DINOv2 ViT-S/14 (~85 MB) from torch.hub — do this before any demo, on hotel
wifi if you must, never on stage.

## Datasets

Layout expected under `data/`:

```
data/mendeley/{healthy,crack,squat,spalling,flaking,shelling,groove,joint,fastener}/*.jpg
data/rsdds/{healthy,defect}/*.jpg                      (optional, cross-domain)
data/fastener/{healthy_fastener,defective_fastener}/*  (optional, second domain)
```

1. **Mendeley Railway Track Surface Faults** (primary, 7 defect classes):
   https://data.mendeley.com/datasets/8hxtgyyxrw/2 — open download, unzip class folders into `data/mendeley/`.
   Folder names must be lowercase as above (rename `Cracks` → `crack`, etc.).
   ⚠ This dataset has **no healthy class** — and its frames are consecutive video stills, so the
   perceptual-hash dedup matters; never split it randomly.
2. **Healthy / binary domain** — RTFD (GitHub mirror of the Kaggle salmaneunus dataset, no account needed):
   https://github.com/fangvv/RTFD-Dataset → `RailwayDefectDetectionDatabase V2.zip` →
   "Non defective" images go to `data/mendeley/healthy/` and `data/kaggle/healthy/`,
   "Defective" to `data/kaggle/defective/`.
   Because healthy images come from a different capture domain than the Mendeley defects, calibrate
   the anomaly threshold per domain (the `/initialize` percentile + conformal calibration do this).
3. **RSDDs** (academic benchmark, compare against FS-RSDD's 95.2/99.1 ROC):
   https://ieee-dataport.org/documents/rsdds
4. **Kaggle Fastener subset** (second domain):
   `kaggle datasets download ashikadnan/railway-track-fault-detection-dataset2fastener`

## Run Everything

```bash
# 1. Train the healthy-only memory bank (stage 1)
python train.py --data-dir ./data/mendeley --mode healthy

# 2. Build per-image class embeddings (top-K anomalous patches per image)
python train.py --data-dir ./data/mendeley --mode embed

# 3. Few-shot evaluation: closed-set + open-set, mean ± std over 100 episodes
python train.py --mode few-shot --n-shots 5 --n-ways 5 --n-episodes 100

# 4. Ablation vs transfer-learning baselines (ResNet-18 ImageNet rows)
python train.py --data-dir ./data/mendeley --mode benchmark

# Or the full experiment suite in one shot (incl. conformal + cross-domain):
python run_experiments.py --data-dir ./data/mendeley \
    --rsdds-dir ./data/rsdds --fastener-dir ./data/fastener --ablation
```

Results land in `results/experiment_results.json` and `models/`.

## Run the Demo (API + Dashboard)

```bash
# Terminal 1 — API on :8000 (loads embeddings produced by train.py)
python run_api.py

# Terminal 2 — dashboard on :3000
cd frontend && npm run dev
```

Then initialize once (uses the stored healthy embeddings):

```bash
curl -X POST localhost:8000/initialize -H 'Content-Type: application/json' -d '{"threshold_percentile": 95}'
curl -X POST localhost:8000/few-shot/setup -F classes=crack -F classes=squat -F classes=spalling -F n_shots=5
```

### Demo script (the discovery flywheel — practice this)

1. **Upload page**: drop a healthy image → green "healthy" + heatmap. Drop a crack → "defect_detected",
   severity, failure risk, anomaly heatmap, DINOv2 attention.
2. **Upload a defect type the prototypes don't know** (e.g. a groove when only crack/squat/spalling are
   set up) → "unknown_anomaly".
3. **Discovery page**: the unknown is there with its thumbnail. Click *Label*, type `groove`.
4. **Upload another groove** → now classified as `groove`. *That's "adapts to new defect types in
   minutes, no retraining" demonstrated live.*
5. **Maintenance page**: every detection populated the digital twin — priorities ranked by failure risk.
6. **Few-Shot Lab**: run 5-way 5-shot, 100 episodes → mean ± CI + unknown-detection rate, live.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | liveness + model state |
| `/initialize` | POST | fit PatchCore from stored healthy embeddings |
| `/train/healthy` | POST (multipart, multiple `files`) | build memory bank from uploaded healthy images |
| `/detect` | POST (file, track_id, location_m) | full two-stage pipeline; returns label, severity, heatmap/attention/mask (base64) |
| `/few-shot/setup` | POST | build prototypes from stored class embeddings |
| `/evaluate/few-shot` | POST | episodic eval (closed + open-set) |
| `/discovery/unknowns` | GET | open-set rejections awaiting labels |
| `/discovery/label` | POST | label an unknown → instantly a few-shot class |
| `/stats`, `/stats/defect-distribution` | GET | live pipeline metrics for the dashboard |
| `/digital-twin/status[/{track_id}]` | GET | segment health, priority rankings |

## Evaluation Methodology (what makes the numbers defensible)

- **Episodes, not single runs**: mean ± std and 95% CI over ≥100 random support draws.
- **Strict K-shot**: classes with < K+1 images are excluded, never silently reduced.
- **Open-set protocol**: each episode holds out an extra class as a true unknown; we report unknown
  detection rate *and* false-unknown rate on knowns.
- **No leakage**: one embedding per image; perceptual-hash dedup before any split.
- **Same protocol for every ablation row** (ResNet-18 centroid / linear probe / DINOv2 variants).
- **Conformal operating point**: threshold from a calibration split of defect scores with a
  finite-sample recall guarantee, evaluated on a held-out split.

## Tests

```bash
pytest tests/ -q   # 22 tests: sampler leakage, open-set, coreset, conformal guarantee, twin
```

## References

- AnomalyDINO: patch-based few-shot anomaly detection with DINOv2 (WACV 2025) — arXiv:2405.14529
- PatchCore: Towards Total Recall in Industrial Anomaly Detection (CVPR 2022) — arXiv:2106.08265
- Prototypical Networks for Few-shot Learning (NeurIPS 2017) — arXiv:1703.05175
- FS-RSDD: Few-Shot Rail Surface Defect Detection (Sensors 2023) — the published benchmark to compare against
- Conformal prediction for railway signaling — arXiv:2304.06052
- DINOv2: Learning Robust Visual Features without Supervision — arXiv:2304.07193
