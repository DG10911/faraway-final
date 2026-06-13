import base64
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
    all_embs = {k: v for k, v in system.embedding_db.get_all().items() if k != "healthy_tokens"}
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
