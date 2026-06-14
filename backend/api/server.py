import base64
import json
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..pipeline.orchestrator import RailGuardFSL

app = FastAPI(title="RailGuard-FSL++ API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = RailGuardFSL()

IMAGE_KEYS = ["heatmap", "attention", "gradcam", "anomaly_mask"]

DEFAULT_FEWSHOT_CLASSES = ["crack", "squat", "spalling", "shelling", "flaking"]


@app.on_event("startup")
def _auto_initialize():
    """Boot straight into a ready state if embeddings already exist on disk
    (e.g. data/embeddings from a prior train run or a shared dataset) — so a
    fresh `python run_api.py` needs no manual /initialize or /few-shot/setup."""
    try:
        # A demo seed (scripts/seed_demo_embeddings.py) may persist calibrated
        # thresholds; prefer them over the percentile defaults, which flag
        # everything when only a handful of healthy frames are available.
        calib = {}
        calib_file = system.embedding_db.storage_path / "calibration.json"
        if calib_file.exists():
            try:
                calib = json.loads(calib_file.read_text())
            except Exception as e:
                print(f"[startup] could not read calibration.json: {e}")

        healthy = system.embedding_db.get("healthy_tokens")
        if healthy is None:
            healthy = system.embedding_db.get("healthy")
        if healthy is not None:
            system.initialize(healthy, 95.0)
            if calib.get("anomaly_threshold") is not None:
                system.anomaly_detector.anomaly_threshold = float(calib["anomaly_threshold"])
                print(f"[startup] applied calibrated anomaly threshold {calib['anomaly_threshold']:.4f}")
            print(f"[startup] auto-initialized PatchCore on {len(healthy)} healthy embeddings")
        support = {c: system.embedding_db.get(c) for c in DEFAULT_FEWSHOT_CLASSES}
        # Keep any class with at least a couple of support shots. The synthetic
        # bank has ~60/class, but a demo-seeded bank (see
        # scripts/seed_demo_embeddings.py) may only have 2-4 real frames/class.
        support = {k: v for k, v in support.items() if v is not None and len(v) >= 2}
        if len(support) >= 2:
            # Use a true 5-shot prototype when every class has the shots for it,
            # otherwise fall back to "use every available shot" so a small
            # demo-seeded bank still boots its prototypes.
            n_shots = 5 if all(len(v) >= 5 for v in support.values()) else None
            system.setup_few_shot(support, n_shots=n_shots)
            # Apply the calibrated open-set rejection threshold so held-out
            # classes surface as unknown_anomaly on the Discovery page (setup_few_shot
            # resets it to None, so this must run after).
            if calib.get("open_set_threshold") is not None and system.few_shot_classifier is not None:
                system.few_shot_classifier.open_set_threshold = float(calib["open_set_threshold"])
                print(f"[startup] applied open-set threshold {calib['open_set_threshold']:.4f}")
            print(f"[startup] auto-setup few-shot classes: {list(support.keys())} (n_shots={n_shots})")
    except Exception as e:  # never block server boot on auto-init
        print(f"[startup] auto-initialize skipped: {e}")


def _decode_upload(contents: bytes) -> np.ndarray:
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _encode_images(result: dict) -> dict:
    for key in IMAGE_KEYS:
        value = result.pop(key, None)
        if isinstance(value, np.ndarray):
            if value.ndim == 3:
                value = cv2.cvtColor(value, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode(".png", value)
            result[f"{key}_b64"] = base64.b64encode(buffer).decode("utf-8")
    result.pop("anomaly_map", None)
    return result


class InitRequest(BaseModel):
    threshold_percentile: float = 95.0


class EvaluateRequest(BaseModel):
    n_ways: int = 5
    n_shots: int = 5
    n_episodes: int = 100
    open_set: bool = True


class LabelRequest(BaseModel):
    unknown_id: str
    label: str


class DefectReport(BaseModel):
    track_id: str = "Track_A12"
    location_m: float = 0.0


class ConformalRequest(BaseModel):
    target_recall: float = 0.95


class ThresholdRequest(BaseModel):
    percentile: float = 95.0
    domain: str = "default"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "RailGuard-FSL++",
        "initialized": system.initialized,
        "few_shot_ready": system.few_shot_classifier is not None,
    }


@app.post("/initialize")
def initialize(req: InitRequest):
    """Initialize PatchCore from healthy embeddings already in the database
    (e.g., produced by `python train.py --mode healthy`)."""
    healthy = system.embedding_db.get("healthy_tokens")
    if healthy is None:
        healthy = system.embedding_db.get("healthy")
    if healthy is None:
        raise HTTPException(400, "No healthy embeddings found. POST /train/healthy or run train.py first.")
    system.initialize(healthy, req.threshold_percentile)
    return {"status": "initialized", "n_embeddings": len(healthy), "threshold": system.anomaly_detector.anomaly_threshold}


@app.post("/train/healthy")
async def train_healthy(files: List[UploadFile] = File(...)):
    """Train the healthy-only memory bank from one or more healthy rail images."""
    all_tokens, all_cls, n_patches = [], [], 0
    for file in files:
        image = _decode_upload(await file.read())
        features = system.embed_image(image)
        if features is None:
            continue
        all_tokens.append(features["tokens"].reshape(-1, features["tokens"].shape[-1]))
        all_cls.append(features["cls"])
        n_patches += len(features["patch_coords"])
    if not all_tokens:
        raise HTTPException(400, "No valid rail regions found in uploaded images")
    tokens = np.concatenate(all_tokens, axis=0)
    cls = np.concatenate(all_cls, axis=0)
    system.embedding_db.store("healthy_tokens", tokens)
    system.embedding_db.store("healthy", cls)
    system.initialize(tokens)
    return {"status": "healthy_model_trained", "n_images": len(files), "n_patches": n_patches, "n_embeddings": len(tokens)}


@app.post("/detect")
async def detect(file: UploadFile = File(...), track_id: str = Form("unknown"), location_m: float = Form(0.0)):
    image = _decode_upload(await file.read())
    result = system.process_frame(image, track_id, location_m)
    return _encode_images(result)


@app.post("/few-shot/setup")
async def setup_few_shot(classes: List[str] = Form(...), n_shots: int = Form(5)):
    support_embeddings = {}
    for cls_name in classes:
        embs = system.embedding_db.get(cls_name)
        if embs is not None:
            support_embeddings[cls_name] = embs
    if not support_embeddings:
        raise HTTPException(400, "No embeddings found for specified classes")
    success = system.setup_few_shot(support_embeddings, n_shots)
    return {"success": success, "n_classes": len(support_embeddings), "n_shots": n_shots}


@app.post("/evaluate/few-shot")
def evaluate_few_shot(req: EvaluateRequest):
    all_embs = {k: v for k, v in system.embedding_db.get_all().items() if k not in ("healthy_tokens", "healthy")}
    if not all_embs:
        raise HTTPException(400, "No embeddings in database — run `python train.py --mode embed` first")
    try:
        return system.evaluate_few_shot_performance(all_embs, req.n_ways, req.n_shots, req.n_episodes, open_set=req.open_set)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/discovery/unknowns")
def discovery_unknowns():
    return {"unknowns": system.get_unknowns()}


@app.post("/discovery/label")
def discovery_label(req: LabelRequest):
    if not system.label_unknown(req.unknown_id, req.label):
        raise HTTPException(404, f"Unknown sample '{req.unknown_id}' not found")
    return {"status": "labeled", "label": req.label, "few_shot_classes": system.embedding_db.list_classes()}


@app.post("/conformal/calibrate")
def conformal_calibrate(req: ConformalRequest):
    """Set a conformal operating point with a finite-sample recall guarantee."""
    try:
        return system.calibrate_conformal(req.target_recall)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/conformal/status")
def conformal_status():
    return {
        "conformal": system.conformal_state,
        "scores": {"n_defect": len(system._defect_scores), "n_healthy": len(system._healthy_scores_obs)},
    }


@app.post("/calibrate/threshold")
def calibrate_threshold(req: ThresholdRequest):
    """Cross-domain calibration: re-pick the anomaly threshold at a healthy percentile."""
    try:
        return system.calibrate_threshold(req.percentile, req.domain)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/calibrate/domains")
def calibrate_domains():
    return {"domains": list(system.domains.values())}


@app.post("/augment/preview")
async def augment_preview(file: UploadFile = File(...), kind: str = Form("crack")):
    """Render a synthetic defect on the uploaded healthy crop (CutPaste/procedural)."""
    image = _decode_upload(await file.read())
    result = system.augment_preview(image, kind)
    out = {"kind": result["kind"]}
    for key in ("original", "synthetic"):
        img = result[key]
        if isinstance(img, np.ndarray):
            _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            out[f"{key}_b64"] = base64.b64encode(buf).decode("utf-8")
    return out


@app.get("/stats")
def stats():
    return system.get_stats()


@app.get("/stats/defect-distribution")
def defect_distribution():
    return {"distribution": system.get_defect_distribution()}


@app.get("/digital-twin/status")
def twin_overview():
    segments = [system.twin_mgr.get_segment_status(tid) for tid in system.twin_mgr.segments]
    return {"segments": segments}


@app.get("/digital-twin/status/{track_id}")
def twin_status(track_id: str):
    return system.twin_mgr.get_segment_status(track_id)


@app.post("/digital-twin/report")
def report_to_twin(report: DefectReport):
    return system.twin_mgr.get_segment_status(report.track_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
