#!/usr/bin/env python3
"""Seed the embedding DB from the REAL demo gallery images.

Why this exists
---------------
By default the few-shot prototypes are built from the *synthetic* textures in
``data/mendeley`` (see ``scripts/make_synthetic_data.py``). The dashboard demo
gallery, however, serves real video frames in ``frontend/public/demo/*.jpg``.
Those two distributions are far apart in DINOv2 space, so a real demo crack gets
matched to the nearest *synthetic* prototype -- often ``squat`` or ``spalling``.

This script rebuilds the healthy memory bank and the few-shot support set from
the demo images themselves, so the prototypes live in the SAME distribution the
gallery shows. After running it and restarting the API, clicking a demo image
classifies to its own label.

Usage (from the repo root, inside the venv that has torch)::

    python scripts/seed_demo_embeddings.py
    python run_api.py          # restart so the API auto-loads the new embeddings

``groove`` and ``joint`` are intentionally left OUT of the few-shot set so the
Discovery "label an unknown" flow still has a genuine novel class to find.
"""
import json
import os
import sys

import cv2
import numpy as np

# allow ``python scripts/seed_demo_embeddings.py`` from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.pipeline.orchestrator import RailGuardFSL  # noqa: E402

DEMO_DIR = os.path.join("frontend", "public", "demo")
# defect classes the gallery should classify to (must match the API's
# DEFAULT_FEWSHOT_CLASSES so a plain `python run_api.py` boots them up)
KNOWN_DEFECTS = ["crack", "squat", "spalling", "shelling", "flaking"]
MIN_PER_CLASS = 2  # guarantee enough support shots even from 2 demo frames
# With only a few healthy demo frames, the 95th-percentile-of-tokens threshold
# is far too sensitive (every frame, even healthy, trips it). Instead we anchor
# the threshold just above the worst healthy FRAME score. Healthy demo frames
# peak ~0.32 and every defect is >=0.64, so a 1.3x margin sits cleanly between.
HEALTHY_THRESHOLD_MARGIN = 1.3
# Open-set rejection: anything farther from every known prototype than the worst
# genuine known frame (times this margin) is reported as `unknown_anomaly`. This
# is what makes a held-out class (groove/joint) surface on the Discovery page
# instead of being force-fit to the nearest known defect.
OPEN_SET_MARGIN = 1.02
# Persisted next to the embeddings so the API reloads it on `python run_api.py`.
CALIB_FILE = "calibration.json"


def load_groups():
    """{label: [image_path, ...]} from the gallery manifest (with a fallback
    to whatever crack_*/healthy_* files are actually on disk)."""
    manifest_path = os.path.join(DEMO_DIR, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = []
        for fname in sorted(os.listdir(DEMO_DIR)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                manifest.append({"src": f"/demo/{fname}", "label": fname.split("_")[0]})
    groups = {}
    for item in manifest:
        path = os.path.join("frontend", "public", item["src"].lstrip("/"))
        if os.path.exists(path):
            groups.setdefault(item["label"], []).append(path)
    return groups


def read_rgb(path):
    bgr = cv2.imread(path)
    return None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def defect_region_embeddings(system, paths):
    """CLS embeddings of the most-anomalous patches across a class's frames --
    this mirrors what process_frame() classifies at inference time."""
    embs = []
    for p in paths:
        feat = system.embed_image(read_rgb(p))
        if feat is None:
            print(f"  ! no rail region: {os.path.basename(p)}")
            continue
        res = system.anomaly_detector.predict(
            feat["tokens"], feat["patch_coords"], feat["rail_region"].shape[:2]
        )
        scores = np.asarray(res["patch_scores"])
        idx = np.where(scores > res["threshold"])[0]
        if len(idx) < MIN_PER_CLASS:  # fall back to the top-scoring patches
            idx = np.argsort(scores)[-min(MIN_PER_CLASS, len(scores)):]
        embs.append(feat["cls"][idx])
    return np.concatenate(embs, axis=0) if embs else None


def frame_nearest_distance(system, clf, path):
    """Smallest distance from any anomalous patch of `path` to a known
    prototype -- i.e. how 'known' the frame looks to the few-shot classifier."""
    feat = system.embed_image(read_rgb(path))
    if feat is None:
        return None
    res = system.anomaly_detector.predict(
        feat["tokens"], feat["patch_coords"], feat["rail_region"].shape[:2]
    )
    scores = np.asarray(res["patch_scores"])
    idx = np.where(scores > res["threshold"])[0]
    if len(idx) == 0:
        idx = np.array([int(np.argmax(scores))])
    distances, _ = clf._distances(feat["cls"][idx])
    return float(np.min(distances))


def main():
    if not os.path.isdir(DEMO_DIR):
        sys.exit(f"Demo dir not found: {DEMO_DIR} (run from the repo root)")

    system = RailGuardFSL()
    groups = load_groups()
    print(f"Demo classes found: {sorted(groups)}\n")

    # 1) Healthy memory bank (PatchCore) from the healthy demo frames ---------
    healthy_tokens, healthy_cls = [], []
    for p in groups.get("healthy", []):
        feat = system.embed_image(read_rgb(p))
        if feat is None:
            print(f"  ! healthy frame had no rail region: {os.path.basename(p)}")
            continue
        healthy_tokens.append(feat["tokens"].reshape(-1, feat["tokens"].shape[-1]))
        healthy_cls.append(feat["cls"])
    if not healthy_tokens:
        sys.exit("No healthy demo frames produced a rail region -- aborting.")
    tokens = np.concatenate(healthy_tokens, axis=0)
    system.embedding_db.store("healthy_tokens", tokens)
    system.embedding_db.store("healthy", np.concatenate(healthy_cls, axis=0))
    system.initialize(tokens, 95.0)

    # Anchor the anomaly threshold just above the worst healthy *frame* score so
    # healthy frames pass while every defect (>=0.64) still trips it.
    healthy_frame_max = max(
        system.anomaly_detector.predict(
            f["tokens"], f["patch_coords"], f["rail_region"].shape[:2]
        )["anomaly_score"]
        for f in (system.embed_image(read_rgb(p)) for p in groups.get("healthy", []))
    )
    threshold = float(healthy_frame_max * HEALTHY_THRESHOLD_MARGIN)
    system.anomaly_detector.anomaly_threshold = threshold
    # calibration.json is written once at the end, with the open-set threshold too
    print(f"[healthy] memory bank: {len(tokens)} patch tokens "
          f"from {len(groups.get('healthy', []))} frames "
          f"(healthy_frame_max={healthy_frame_max:.4f}, threshold={threshold:.4f})\n")

    # 2) Per-class few-shot support from the demo defect regions --------------
    for label in KNOWN_DEFECTS:
        paths = groups.get(label, [])
        if not paths:
            print(f"[{label}] no demo frames -- skipped")
            continue
        embs = defect_region_embeddings(system, paths)
        if embs is None or len(embs) == 0:
            print(f"[{label}] no usable embeddings -- skipped")
            continue
        system.embedding_db.store(label, embs)
        print(f"[{label}] {len(embs)} defect-region embeddings from {len(paths)} frames")

    # 3) Build the few-shot prototypes (use every stored shot) ----------------
    support = {c: system.embedding_db.get(c) for c in KNOWN_DEFECTS}
    support = {k: v for k, v in support.items() if v is not None}
    system.setup_few_shot(support, n_shots=None)
    print(f"\n[few-shot] prototypes built for {list(support.keys())}")

    # 3b) Open-set threshold: just above the worst genuine known frame, so a
    # held-out class (groove/joint) is rejected as `unknown_anomaly` and surfaces
    # on the Discovery page instead of being forced onto a known prototype.
    clf = system.few_shot_classifier
    known_dists = [
        d for label in KNOWN_DEFECTS for d in
        (frame_nearest_distance(system, clf, p) for p in groups.get(label, []))
        if d is not None
    ]
    open_set_threshold = float(max(known_dists) * OPEN_SET_MARGIN) if known_dists else None
    if open_set_threshold is not None:
        clf.open_set_threshold = open_set_threshold
        print(f"[open-set] rejection threshold = {open_set_threshold:.4f} "
              f"(worst known frame distance {max(known_dists):.4f})")

    # Persist both calibrated thresholds for the API to reload on startup.
    calib = {"anomaly_threshold": threshold}
    if open_set_threshold is not None:
        calib["open_set_threshold"] = open_set_threshold
    (system.embedding_db.storage_path / CALIB_FILE).write_text(json.dumps(calib))

    # 4) Self-check: classify every demo frame and report predicted vs label --
    print("\n--- self-check (does each demo frame classify to its label?) ---")
    hits = total = 0
    for label in sorted(groups):
        for p in groups[label]:
            out = system.process_frame(read_rgb(p), "selfcheck", 0.0)
            pred = out.get("label") or out.get("status")
            expected_unknown = label in ("groove", "joint")  # left out on purpose
            ok = (pred == label) or (label == "healthy" and out.get("status") == "healthy") \
                or (expected_unknown and out.get("status") in ("unknown_anomaly",
                                                               "anomaly_detected_unclassified"))
            hits += ok
            total += 1
            mark = "OK " if ok else "XX "
            print(f"  {mark} {os.path.basename(p):18s} label={label:9s} -> {pred}")
    print(f"\n{hits}/{total} frames behaved as expected.")
    print("Now restart the API so it loads the new embeddings:  python run_api.py")


if __name__ == "__main__":
    main()
