import base64
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch

from .phase1_input_validation import InputValidator
from .phase2_rail_extraction import RailExtractor
from .phase3_patch_pipeline import PatchPipeline
from ..models.dinov2_encoder import DINOv2Encoder
from ..models.patchcore_anomaly import PatchCoreDetector
from ..models.prototypical_network import PrototypicalNetwork, evaluate_few_shot
from ..models.open_set_recognizer import OpenSetRecognizer
from ..models.severity_estimator import SeverityEstimator
from ..utils.embeddings_db import EmbeddingDatabase
from ..utils.hard_negatives import HardNegativeManager
from ..utils.explainability import ExplainabilityEngine
from ..twins.digital_twin import DigitalTwinManager
from ..evaluation.metrics import Evaluator


class RailGuardFSL:
    """Two-stage pipeline: healthy-only PatchCore screening on DINOv2 patch
    tokens, then few-shot prototypical classification (with open-set
    rejection) on the CLS embeddings of anomalous patches only."""

    def __init__(self, device: Optional[str] = None):
        from ..utils.device import resolve_device
        self.device = resolve_device(device)
        self.validator = InputValidator()
        self.extractor = RailExtractor()
        self.patcher = PatchPipeline()
        self.encoder = DINOv2Encoder(device=self.device)
        self.anomaly_detector = PatchCoreDetector(device=self.device)
        self.few_shot_classifier = None
        self.open_set_recognizer = OpenSetRecognizer()
        self.severity_estimator = SeverityEstimator()
        self.embedding_db = EmbeddingDatabase()
        self.hard_negative_mgr = HardNegativeManager()
        self.explainability = ExplainabilityEngine(device=self.device)
        self.twin_mgr = DigitalTwinManager()
        self.evaluator = Evaluator()
        self.initialized = False

        self.unknown_samples: List[Dict] = []
        self.stats = {
            "frames_processed": 0,
            "frames_rejected": 0,
            "anomalies_found": 0,
            "defects_confirmed": 0,
            "unknowns_flagged": 0,
        }
        self._latencies = deque(maxlen=50)
        self.alerts = deque(maxlen=20)

        # operating-point calibration state
        self._defect_scores = deque(maxlen=500)    # anomaly scores of frames flagged anomalous
        self._healthy_scores_obs = deque(maxlen=2000)  # anomaly scores of frames passed as healthy
        self.conformal_state: Optional[Dict] = None
        self.domains: Dict[str, Dict] = {}

    # ------------------------------------------------------------------ setup

    def initialize(self, healthy_features: np.ndarray, threshold_percentile: float = 95.0):
        """healthy_features: (n, dim) CLS features or (n, tokens, dim) /
        flattened token features from healthy rail only."""
        self.anomaly_detector.fit(healthy_features, threshold_percentile)
        self.initialized = True

    def embed_image(self, image: np.ndarray) -> Optional[Dict]:
        """Validation -> rail crop -> patches -> CLS + token features."""
        validation = self.validator.validate(image)
        if validation["status"] != "valid":
            return None
        rail_region = self.extractor.extract_rail_region(image)
        if rail_region is None:
            return None
        patches = self.patcher.extract_patches(rail_region)
        patch_images = [p for p, _ in patches]
        return {
            "rail_region": rail_region,
            "patch_coords": [c for _, c in patches],
            "cls": self.encoder.embed_patches(patch_images),
            "tokens": self.encoder.embed_patch_tokens(patch_images),
        }

    # -------------------------------------------------------------- inference

    def process_frame(self, image: np.ndarray, track_id: str = "unknown", location_m: float = 0.0) -> Dict:
        start = time.time()
        self.stats["frames_processed"] += 1

        validation = self.validator.validate(image)
        if validation["status"] != "valid":
            self.stats["frames_rejected"] += 1
            return validation

        rail_region = self.extractor.extract_rail_region(image)
        if rail_region is None:
            self.stats["frames_rejected"] += 1
            return {"status": "error", "reason": "rail_extraction_failed"}

        patches = self.patcher.extract_patches(rail_region)
        patch_images = [p for p, _ in patches]
        patch_coords = [c for _, c in patches]

        token_features = self.encoder.embed_patch_tokens(patch_images)

        if not self.initialized:
            return {"status": "not_initialized", "patches_extracted": len(patches)}

        anomaly_result = self.anomaly_detector.predict(token_features, patch_coords, rail_region.shape[:2])
        # record the raw anomaly score for operating-point / conformal calibration
        (self._defect_scores if anomaly_result["is_anomaly"] else self._healthy_scores_obs).append(
            float(anomaly_result["anomaly_score"])
        )
        heatmap = self.explainability.generate_anomaly_heatmap(rail_region, anomaly_result["anomaly_map"])
        attention = self.explainability.generate_attention_map(self.encoder, rail_region)
        self._latencies.append((time.time() - start) * 1000)

        base = {
            "anomaly_score": anomaly_result["anomaly_score"],
            "anomaly_threshold": anomaly_result["threshold"],
            "heatmap": heatmap,
            "attention": attention,
            "anomaly_mask": anomaly_result["anomaly_mask"],
        }

        if not anomaly_result["is_anomaly"]:
            return {"status": "healthy", **base}

        self.stats["anomalies_found"] += 1

        # Stage 2 operates only on the patches PatchCore flagged, so a tiny
        # defect is not averaged away by the healthy majority of the frame.
        patch_scores = np.array(anomaly_result["patch_scores"])
        anomalous_idx = np.where(patch_scores > anomaly_result["threshold"])[0]
        if len(anomalous_idx) == 0:
            anomalous_idx = np.array([int(np.argmax(patch_scores))])
        cls_features = self.encoder.embed_patches([patch_images[i] for i in anomalous_idx])

        hard_neg_flags = [self.hard_negative_mgr.has_hard_negative(emb) for emb in cls_features]
        if hard_neg_flags and all(hard_neg_flags):
            return {"status": "healthy", "note": "hard_negative_suppressed", **base}

        if self.few_shot_classifier is None or self.few_shot_classifier.prototypes is None:
            severity = self.severity_estimator.estimate(
                anomaly_result["anomaly_score"], anomaly_result["anomaly_mask"], 0.5
            )
            self._record_unknown(rail_region, cls_features, track_id, anomaly_result["anomaly_score"], distance=None)
            return {"status": "anomaly_detected_unclassified", "severity": severity, **base}

        results = self.few_shot_classifier.open_set_classify(cls_features)
        known = [r for r in results if r["label"] != "unknown_anomaly"]

        if not known:
            primary = max(results, key=lambda r: r["confidence"])
            severity = self.severity_estimator.estimate(
                anomaly_result["anomaly_score"], anomaly_result["anomaly_mask"], primary["confidence"]
            )
            self._record_unknown(rail_region, cls_features, track_id, anomaly_result["anomaly_score"], primary["distance"])
            self._add_alert(f"Unknown anomaly on {track_id} at {location_m:.0f}m")
            return {
                "status": "unknown_anomaly",
                "label": "unknown_anomaly",
                "confidence": primary["confidence"],
                "severity": severity,
                **base,
            }

        # Majority vote across anomalous patches; primary = highest confidence.
        labels = [r["label"] for r in known]
        label = max(set(labels), key=labels.count)
        confidence = float(np.mean([r["confidence"] for r in known if r["label"] == label]))

        severity = self.severity_estimator.estimate(
            anomaly_result["anomaly_score"], anomaly_result["anomaly_mask"], confidence
        )
        twin_result = self.twin_mgr.report_defect(
            track_id=track_id,
            location_m=location_m,
            defect_type=label,
            severity=severity["severity"],
            anomaly_score=anomaly_result["anomaly_score"],
            confidence=confidence,
        )
        self.stats["defects_confirmed"] += 1
        self._add_alert(f"{label.capitalize()} detected on {track_id} at {location_m:.0f}m ({severity['severity']})")

        return {
            "status": "defect_detected",
            "label": label,
            "confidence": confidence,
            "severity": severity,
            "patch_coords": [patch_coords[i] for i in anomalous_idx],
            "failure_risk": twin_result["failure_risk_pct"],
            "event_id": twin_result["event_id"],
            "priority_rank": twin_result["priority_rankings"][0]["priority_rank"] if twin_result["priority_rankings"] else "N/A",
            **base,
        }

    # ------------------------------------------------------------- few-shot

    def setup_few_shot(self, support_embeddings: Dict[str, np.ndarray], n_shots: Optional[int] = 5):
        """n_shots=None uses every available embedding per class (used when
        incrementally labeling discovered unknowns)."""
        all_embs, all_labels, class_names = [], [], []
        for class_name, embs in support_embeddings.items():
            embs = np.asarray(embs)
            if len(embs) == 0:
                continue
            take = len(embs) if n_shots is None else min(n_shots, len(embs))
            if n_shots is not None and len(embs) < n_shots:
                continue
            label_idx = len(class_names)
            all_embs.append(embs[:take])
            all_labels.extend([label_idx] * take)
            class_names.append(class_name)
        if not all_embs:
            return False
        all_embs = np.concatenate(all_embs, axis=0)
        self.few_shot_classifier = PrototypicalNetwork(
            embedding_dim=all_embs.shape[1],
            n_ways=len(class_names),
            n_shots=n_shots or 0,
        )
        self.few_shot_classifier.compute_prototypes(all_embs, np.array(all_labels), class_names)
        return True

    def evaluate_few_shot_performance(self, all_embeddings: Dict[str, np.ndarray], n_ways: int = 5, n_shots: int = 5, n_episodes: int = 100, open_set: bool = False):
        return evaluate_few_shot(all_embeddings, n_ways, n_shots, n_episodes, open_set=open_set)

    # ------------------------------------------------------------- discovery

    def _record_unknown(self, rail_region: np.ndarray, cls_features: np.ndarray, track_id: str, anomaly_score: float, distance: Optional[float]):
        thumb = cv2.resize(rail_region, (192, max(1, int(192 * rail_region.shape[0] / max(rail_region.shape[1], 1)))))
        ok, buf = cv2.imencode(".png", cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
        self.stats["unknowns_flagged"] += 1
        self.unknown_samples.append({
            "id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "track_id": track_id,
            "anomaly_score": round(float(anomaly_score), 4),
            "distance": round(float(distance), 4) if distance is not None else None,
            "thumbnail_b64": base64.b64encode(buf).decode("utf-8") if ok else None,
            "embedding": cls_features.mean(axis=0),
            "labeled_as": None,
        })

    def label_unknown(self, unknown_id: str, label: str) -> bool:
        """The discovery flywheel: a flagged unknown gets a human label and
        immediately becomes a few-shot support example for that class."""
        sample = next((s for s in self.unknown_samples if s["id"] == unknown_id), None)
        if sample is None:
            return False
        existing = self.embedding_db.get(label)
        emb = sample["embedding"][np.newaxis, :]
        combined = np.concatenate([existing, emb], axis=0) if existing is not None else emb
        self.embedding_db.store(label, combined)
        sample["labeled_as"] = label
        # Rebuild prototypes over the defect classes only (never the healthy
        # banks), and keep the calibrated open-set threshold that setup_few_shot
        # would otherwise reset to None.
        prev_threshold = self.few_shot_classifier.open_set_threshold if self.few_shot_classifier else None
        support = {
            k: v for k, v in self.embedding_db.get_all().items()
            if k not in ("healthy_tokens", "healthy") and not k.startswith("_")
        }
        self.setup_few_shot(support, n_shots=None)
        if prev_threshold is not None and self.few_shot_classifier is not None:
            self.few_shot_classifier.open_set_threshold = prev_threshold
        return True

    def get_unknowns(self) -> List[Dict]:
        return [
            {k: v for k, v in s.items() if k != "embedding"}
            for s in self.unknown_samples
            if s["labeled_as"] is None
        ]

    # ------------------------------------------------------- calibration / safety

    def calibrate_conformal(self, target_recall: float = 0.95) -> Dict:
        """Set a conformal operating point with a finite-sample recall guarantee
        from the anomaly scores observed so far (defects = anomalous frames)."""
        from ..utils.conformal import conformal_recall_threshold, evaluate_operating_point
        defect = np.array(self._defect_scores, dtype=np.float64)
        healthy = np.array(self._healthy_scores_obs, dtype=np.float64)
        if len(defect) == 0:
            raise ValueError("No defect scores observed yet — run a few detections (incl. defects) first.")
        res = conformal_recall_threshold(defect, target_recall)
        op = evaluate_operating_point(healthy, defect, res["threshold"])
        if self.anomaly_detector is not None and self.anomaly_detector.anomaly_threshold is not None:
            self.anomaly_detector.anomaly_threshold = res["threshold"]
            res["applied"] = True
        self.conformal_state = {
            **res,
            "empirical_recall": op["empirical_recall"],
            "false_positive_rate": op["false_positive_rate"],
            "n_defect": op["n_defect"],
            "n_healthy": op["n_healthy"],
        }
        return self.conformal_state

    def calibrate_threshold(self, percentile: float, domain: Optional[str] = None) -> Dict:
        """Cross-domain calibration: re-pick the anomaly threshold at a healthy
        percentile (per capture domain) without rebuilding the memory bank."""
        from ..utils.conformal import evaluate_operating_point
        if self.anomaly_detector is None or self.anomaly_detector.healthy_scores is None:
            raise ValueError("Detector not initialized — run /initialize first.")
        threshold = self.anomaly_detector.recalibrate(percentile)
        healthy = np.asarray(self.anomaly_detector.healthy_scores)
        defect = np.array(self._defect_scores, dtype=np.float64)
        op = evaluate_operating_point(healthy, defect, threshold)
        record = {
            "domain": domain or "default",
            "percentile": percentile,
            "threshold": float(threshold),
            "false_positive_rate": op["false_positive_rate"],
            "empirical_recall": op["empirical_recall"],
            "n_defect": op["n_defect"],
        }
        self.domains[record["domain"]] = record
        return record

    def augment_preview(self, image: np.ndarray, kind: str = "crack") -> Dict:
        """Render a synthetic defect on a healthy crop for the augmentation demo."""
        from ..datasets.synthetic_augment import synthesize_defect
        rail = self.extractor.extract_rail_region(image)
        base = rail if rail is not None else image
        synth = synthesize_defect(base, kind)
        return {"kind": kind, "original": base, "synthetic": synth}

    # ----------------------------------------------------------------- stats

    def _add_alert(self, message: str):
        self.alerts.appendleft({"message": message, "timestamp": datetime.now().strftime("%H:%M:%S")})

    def get_stats(self) -> Dict:
        avg_latency = float(np.mean(self._latencies)) if self._latencies else None
        return {
            **self.stats,
            "avg_latency_ms": round(avg_latency, 1) if avg_latency else None,
            "throughput_fps": round(1000.0 / avg_latency, 2) if avg_latency else None,
            "initialized": self.initialized,
            "few_shot_ready": self.few_shot_classifier is not None,
            "alerts": list(self.alerts),
            "calibration_scores": {"n_defect": len(self._defect_scores), "n_healthy": len(self._healthy_scores_obs)},
            "conformal": self.conformal_state,
        }

    def get_defect_distribution(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for segment in self.twin_mgr.segments.values():
            for defect in segment.defects:
                counts[defect.defect_type] = counts.get(defect.defect_type, 0) + 1
        if self.stats["unknowns_flagged"]:
            counts["unknown"] = counts.get("unknown", 0) + self.stats["unknowns_flagged"]
        return counts
