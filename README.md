# RailGuard-FSL++

**Few-Shot Defect Detection in Rail Infrastructure**

A research-grade AI platform combining self-supervised representation learning, few-shot classification, open-set defect discovery, and predictive maintenance intelligence for railway surface defect detection.

## Core Research Question

> How can railway defects be detected when defect examples are extremely rare and new defect categories continuously appear?

## Architecture

```
Camera/video frames → Frame validity gate (blur + rail-presence check)
  → Rail region crop → Overlapping patches → DINOv2 encoder
    → Stage 1: PatchCore healthy-only anomaly detection
      → (normal → Pass)
      → (anomalous) → Stage 2: Prototypical few-shot classifier
        → (near prototype → Defect type + alert)
        → (far from all → Unknown anomaly — flag for review)
```

## Pipeline Phases

| Phase | Component | Description |
|-------|-----------|-------------|
| 1 | Input Validation | Rail presence, blur, occlusion, quality checks |
| 2 | Rail Extraction | RailSeg segmentation, cropping, perspective correction |
| 3 | Patch Pipeline | 256×256 overlapping patches with coordinate tracking |
| 4 | DINOv2 Features | Self-supervised representation learning |
| 5 | PatchCore Anomaly | Healthy-only memory bank anomaly detection |
| 6 | Few-Shot Classifier | Prototypical networks (1/5/10-shot) with median prototypes |
| 7 | Open-Set Recognition | Unknown anomaly detection via distance + score thresholds |
| 8 | Hard Negative Learning | Welds, fishplates, joints, fasteners, stains, rust suppression |
| 9 | Cross-Domain Adaptation | Train Mendeley → adapt RSDDs/Fastener (5 shots) |
| 10 | Explainability | Grad-CAM, Attention Maps, Anomaly Heatmaps |
| 11 | Severity Estimation | Defect area, density, composite → Low/Med/High/Critical |
| 12 | Digital Twin | Virtual segments, failure risk %, maintenance priority |
| 13 | Dashboard | Next.js + Tailwind monitoring and analysis UI |

## Research Novelty

The key contribution is the combination of:

- **Healthy-Only Anomaly Detection** — PatchCore trained exclusively on healthy rail
- **Few-Shot Classification** — Prototypical networks with median prototypes
- **Open-Set Defect Discovery** — Novel defect categories detected via distance thresholds
- **Cross-Domain Adaptation** — 5-shot adaptation to new rail types
- **Railway Digital Twin** — Predictive maintenance with failure risk estimation

## Datasets

1. **Mendeley Railway Surface Faults** — 7 defect classes (primary dataset)
2. **RSDDs** — IEEE DataPort benchmark (Type-I/Type-II)
3. **Kaggle Railway Track Fault Detection** — Binary baseline
4. **Kaggle Fastener Dataset** — Second domain for cross-domain eval

## Quick Start

```bash
# Backend (Poetry)
poetry install
poetry run python run_api.py

# Or activate the shell:
poetry shell
python run_api.py

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

## CLI Usage

```bash
# Train on healthy images
python train.py --data-dir ./data/mendeley --mode healthy

# Embed all defect classes
python train.py --data-dir ./data/mendeley --mode embed

# Evaluate few-shot performance
python train.py --mode few-shot --n-shots 5 --n-ways 5 --n-episodes 100

# Full experiment suite
python run_experiments.py --data-dir ./data/mendeley --rsdds-dir ./data/rsdds
```

## Evaluation Metrics

- Accuracy, Precision, Recall, F1, mAP, AUROC
- Confusion Matrix
- Few-Shot Accuracy Mean ± Std (100 episodes)
- Cross-Domain Accuracy
- Unknown Defect Detection Rate
- False Positive Rate

## Citation

```bibtex
@software{railguard_fslpp,
  title = {RailGuard-FSL++: Few-Shot Defect Detection in Rail Infrastructure},
  year = {2026},
}
```