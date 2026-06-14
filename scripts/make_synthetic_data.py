"""
Generate a synthetic rail-surface dataset so the WHOLE pipeline + dashboard can
run with zero downloads. Images are deliberately simple but class-distinct, so
DINOv2 + PatchCore + the prototypes produce a believable live demo.

Usage:
    python scripts/make_synthetic_data.py                 # default 60/class into ./data/mendeley
    python scripts/make_synthetic_data.py --per-class 40 --out ./data/mendeley

Then continue with the normal flow:
    python train.py --data-dir ./data/mendeley --mode healthy
    python train.py --data-dir ./data/mendeley --mode embed
    python run_api.py    (and, separately, the frontend)

These images are NOT real rail data — use them only to exercise the system and
the UI. Swap in the real Mendeley/RSDDs sets for reportable numbers.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

H = W = 256
CLASSES = ["healthy", "crack", "squat", "spalling", "flaking", "shelling", "groove", "joint", "fastener"]


def _base_rail(rng: np.random.Generator) -> np.ndarray:
    """Ballast-ish background with two parallel steel rails and a bright railhead."""
    img = rng.normal(95, 18, (H, W)).clip(0, 255)            # ballast/gravel noise
    for cx in (int(W * 0.36), int(W * 0.64)):                # two rails
        img[:, cx - 26:cx + 26] = rng.normal(135, 8, (H, 52)).clip(0, 255)
        img[:, cx - 8:cx + 8] = rng.normal(178, 6, (H, 16)).clip(0, 255)  # shiny railhead
    return img


def _railhead_band(rng):
    cx = int(W * (0.36 if rng.random() < 0.5 else 0.64))
    return cx, slice(cx - 8, cx + 8)


def _add_defect(img: np.ndarray, cls: str, rng: np.random.Generator) -> np.ndarray:
    cx, head = _railhead_band(rng)
    if cls == "healthy":
        pass
    elif cls == "crack":                                     # thin dark jagged line down the head
        y, x = rng.integers(20, H - 80), cx + rng.integers(-6, 6)
        for _ in range(rng.integers(60, 110)):
            img[y:y + 2, x:x + 2] = rng.normal(40, 10)
            y += 1; x += rng.integers(-2, 3)
            x = int(np.clip(x, cx - 8, cx + 8))
    elif cls == "squat":                                     # dark elliptical depression on head
        y, x = rng.integers(40, H - 40), cx
        yy, xx = np.ogrid[:H, :W]
        m = ((yy - y) / rng.integers(10, 18)) ** 2 + ((xx - x) / rng.integers(6, 10)) ** 2 <= 1
        img[m] = rng.normal(55, 12, m.sum())
    elif cls in ("spalling", "flaking", "shelling"):         # irregular bright/dark surface loss
        tone = {"spalling": 60, "flaking": 200, "shelling": 90}[cls]
        for _ in range(rng.integers(5, 12)):
            y, x = rng.integers(20, H - 20), cx + rng.integers(-8, 8)
            r = rng.integers(3, 9)
            yy, xx = np.ogrid[:H, :W]
            img[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = rng.normal(tone, 18)
    elif cls == "groove":                                    # bright longitudinal groove
        img[:, cx - 2:cx + 2] = rng.normal(225, 8, (H, 4))
    elif cls == "joint":                                     # transverse gap (rail joint)
        y = rng.integers(60, H - 60)
        img[y:y + 6, cx - 26:cx + 26] = rng.normal(35, 8, (6, 52))
    elif cls == "fastener":                                  # bolt heads beside the rail
        for sy in rng.integers(30, H - 30, size=2):
            for sx in (cx - 34, cx + 34):
                yy, xx = np.ogrid[:H, :W]
                img[(yy - sy) ** 2 + (xx - sx) ** 2 <= 49] = rng.normal(205, 10)
    return img.clip(0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mendeley")
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    total = 0
    for cls in CLASSES:
        d = out / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            img = _add_defect(_base_rail(rng), cls, rng)
            # Rails are built vertically; transpose so they run horizontally — the
            # frame validator (phase 1) keys on >3 long near-horizontal lines.
            img = np.ascontiguousarray(img.T)
            Image.fromarray(img).convert("RGB").save(d / f"{cls}_{i:03d}.png")
            total += 1
        print(f"  {cls:10s} -> {args.per_class} images")
    print(f"Done. {total} synthetic images under {out}/")
    print("Next: python train.py --data-dir %s --mode healthy" % out)


if __name__ == "__main__":
    main()
