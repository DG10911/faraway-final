"""Synthetic defect augmentation for the scarce-positive regime.

When only a handful of real defect examples exist, we expand the few-shot
support set with *synthetic* defects. Two families are implemented here with a
common interface, both dependency-light (numpy + OpenCV) so they run live:

  - CutPaste:     paste a defect-bearing patch onto a healthy rail at a random
                  location, with feathered (alpha) or Poisson (seamless) blending.
                  This is the classic self-supervised anomaly-synthesis trick
                  (Li et al., CutPaste, CVPR 2021) — cheap and surprisingly strong.
  - Procedural:   draw physically-plausible defects (hairline cracks, squats,
                  spalling, grooves) directly onto a healthy crop.

Heavier generators (DefectGAN / DFMGAN / AnomalyDiffusion) are intentionally
*pluggable behind the same `synthesize(...)` signature*: swap the body of
`synthesize_defect` for a model call and the rest of the pipeline is unchanged.
See the report for the GAN/diffusion landscape.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

DEFECT_KINDS = ("crack", "squat", "spalling", "groove")


# --------------------------------------------------------------------------- #
# Procedural defect drawing
# --------------------------------------------------------------------------- #
def _draw_crack(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    y, x = rng.integers(int(h * 0.1), int(h * 0.5)), rng.integers(int(w * 0.3), int(w * 0.7))
    steps = int(rng.integers(int(h * 0.3), int(h * 0.7)))
    for _ in range(steps):
        tone = int(rng.integers(25, 60))
        cv2.line(out, (x, y), (x + rng.integers(-1, 2), y + 1), (tone, tone, tone), thickness=int(rng.integers(1, 3)))
        x = int(np.clip(x + rng.integers(-2, 3), 0, w - 1))
        y = int(np.clip(y + 1, 0, h - 1))
    return out


def _draw_squat(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    cy, cx = rng.integers(int(h * 0.3), int(h * 0.7)), rng.integers(int(w * 0.3), int(w * 0.7))
    ax, ay = int(rng.integers(6, 14)), int(rng.integers(10, 20))
    overlay = out.copy()
    cv2.ellipse(overlay, (cx, cy), (ax, ay), int(rng.integers(0, 180)), 0, 360, (50, 50, 55), -1)
    return cv2.addWeighted(overlay, 0.6, out, 0.4, 0)


def _draw_spalling(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for _ in range(int(rng.integers(5, 12))):
        cy, cx = rng.integers(0, h), rng.integers(int(w * 0.25), int(w * 0.75))
        r = int(rng.integers(2, 7))
        tone = int(rng.integers(60, 110))
        cv2.circle(out, (cx, cy), r, (tone, tone, tone), -1)
    return out


def _draw_groove(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    x = rng.integers(int(w * 0.35), int(w * 0.65))
    cv2.line(out, (x, 0), (x + rng.integers(-4, 5), h), (220, 220, 225), thickness=int(rng.integers(2, 5)))
    return out


_PROCEDURAL = {"crack": _draw_crack, "squat": _draw_squat, "spalling": _draw_spalling, "groove": _draw_groove}


def synthesize_defect(healthy: np.ndarray, kind: str = "crack", seed: Optional[int] = None) -> np.ndarray:
    """Render a synthetic defect of `kind` onto a healthy rail crop (RGB uint8)."""
    rng = np.random.default_rng(seed)
    kind = kind if kind in _PROCEDURAL else "crack"
    return _PROCEDURAL[kind](np.ascontiguousarray(healthy), rng)


# --------------------------------------------------------------------------- #
# CutPaste
# --------------------------------------------------------------------------- #
def cutpaste(healthy: np.ndarray, defect: np.ndarray, seed: Optional[int] = None,
             seamless: bool = True) -> np.ndarray:
    """Paste a random patch of `defect` onto `healthy` (both RGB uint8)."""
    rng = np.random.default_rng(seed)
    H, W = healthy.shape[:2]
    dh, dw = defect.shape[:2]
    pw, ph = int(rng.integers(max(8, dw // 4), max(12, dw // 2))), int(rng.integers(max(8, dh // 4), max(12, dh // 2)))
    pw, ph = min(pw, dw, W - 2), min(ph, dh, H - 2)
    sx, sy = int(rng.integers(0, dw - pw + 1)), int(rng.integers(0, dh - ph + 1))
    patch = defect[sy:sy + ph, sx:sx + pw]
    tx, ty = int(rng.integers(0, W - pw + 1)), int(rng.integers(0, H - ph + 1))
    out = healthy.copy()
    if seamless:
        try:
            center = (tx + pw // 2, ty + ph // 2)
            mask = np.full(patch.shape[:2], 255, np.uint8)
            return cv2.seamlessClone(patch, out, mask, center, cv2.NORMAL_CLONE)
        except Exception:
            pass
    # feathered alpha fallback
    alpha = np.zeros((ph, pw), np.float32)
    cv2.ellipse(alpha, (pw // 2, ph // 2), (pw // 2, ph // 2), 0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=pw / 6 + 1)[..., None]
    region = out[ty:ty + ph, tx:tx + pw].astype(np.float32)
    out[ty:ty + ph, tx:tx + pw] = (alpha * patch + (1 - alpha) * region).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Batch expansion for few-shot support
# --------------------------------------------------------------------------- #
def expand_support(healthy_crops: List[np.ndarray], n_per_kind: int = 3,
                   kinds: Tuple[str, ...] = DEFECT_KINDS, seed: int = 0) -> Dict[str, List[np.ndarray]]:
    """Grow a synthetic support set: for each defect kind, render `n_per_kind`
    variants over randomly chosen healthy crops. Returns {kind: [imgs]}."""
    rng = np.random.default_rng(seed)
    if not healthy_crops:
        return {}
    out: Dict[str, List[np.ndarray]] = {k: [] for k in kinds}
    for kind in kinds:
        for i in range(n_per_kind):
            base = healthy_crops[int(rng.integers(len(healthy_crops)))]
            out[kind].append(synthesize_defect(base, kind, seed=int(rng.integers(1 << 30))))
    return out
