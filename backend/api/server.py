from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import json
import base64
from typing import Optional
from pydantic import BaseModel
from io import BytesIO
from PIL import Image

from ..pipeline.orchestrator import RailGuardFSL

app = FastAPI(title="RailGuard-FSL++ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = RailGuardFSL()


class InitRequest(BaseModel):
    threshold_percentile: float = 95.0


class FewShotSetup(BaseModel):
    n_shots: int = 5


class EvaluateRequest(BaseModel):
    n_ways: int = 5
    n_shots: int = 5
    n_episodes: int = 100


class DefectReport(BaseModel):
    track_id: str = "Track_A12"
    location_m: float = 0.0


@app.get("/health")
def health():
    return {"status": "ok", "model": "RailGuard-FSL++", "initialized": system.initialized}


@app.post("/initialize")
def initialize(req: InitRequest):
    return {"status": "call_initialize_with_data", "message": "Use POST /train/healthy with healthy images first"}


@app.post("/train/healthy")
async def train_healthy(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rail = system.extractor.extract_rail_region(image)
    if rail is None:
        raise HTTPException(400, "Rail region extraction failed")
    patches = system.patcher.extract_patches(rail)
    patch_imgs = [p for p, _ in patches]
    embs = system.encoder.embed_patches(patch_imgs)
    system.embedding_db.store("healthy", embs, {"source": file.filename})
    system.initialize(embs)
    return {"status": "healthy_model_trained", "n_patches": len(patches), "n_embeddings": len(embs)}


@app.post("/detect")
async def detect(file: UploadFile = File(...), track_id: str = Form("unknown"), location_m: float = Form(0.0)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Invalid image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = system.process_frame(image, track_id, location_m)
    if "heatmap" in result and isinstance(result["heatmap"], np.ndarray):
        _, buffer = cv2.imencode(".png", result["heatmap"])
        result["heatmap_b64"] = base64.b64encode(buffer).decode("utf-8")
        del result["heatmap"]
    if "gradcam" in result and isinstance(result["gradcam"], np.ndarray):
        _, buffer = cv2.imencode(".png", result["gradcam"])
        result["gradcam_b64"] = base64.b64encode(buffer).decode("utf-8")
        del result["gradcam"]
    if "anomaly_mask" in result and isinstance(result["anomaly_mask"], np.ndarray):
        _, buffer = cv2.imencode(".png", result["anomaly_mask"])
        result["anomaly_mask_b64"] = base64.b64encode(buffer).decode("utf-8")
        del result["anomaly_mask"]
    if "anomaly_map" in result and isinstance(result["anomaly_map"], np.ndarray):
        del result["anomaly_map"]
    return result


@app.post("/few-shot/setup")
async def setup_few_shot(classes: list = Form(...), n_shots: int = Form(5)):
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
    all_embs = system.embedding_db.get_all()
    if not all_embs:
        raise HTTPException(400, "No embeddings in database")
    result = system.evaluate_few_shot_performance(all_embs, req.n_ways, req.n_shots, req.n_episodes)
    return result


@app.post("/digital-twin/report")
def report_to_twin(report: DefectReport):
    return system.twin_mgr.get_segment_status(report.track_id)


@app.get("/digital-twin/status/{track_id}")
def twin_status(track_id: str):
    return system.twin_mgr.get_segment_status(track_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
