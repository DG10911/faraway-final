import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .phase1_input_validation import InputValidator
from .phase2_rail_extraction import RailExtractor
from .phase3_patch_pipeline import PatchPipeline
from ..models.dinov2_encoder import DINOv2Encoder
from ..models.patchcore_anomaly import PatchCoreDetector
from ..models.prototypical_network import PrototypicalNetwork, evaluate_few_shot
from ..models.open_set_recognizer import OpenSetRecognizer
from ..models.severity_estimator import SeverityEstimator
from ..utils.embeddings_db import EmbeddingDatabase
from ..utils.hard_negatives import HardNegativeManager, HARD_NEGATIVE_CLASSES
from ..utils.explainability import ExplainabilityEngine
from ..twins.digital_twin import DigitalTwinManager
from ..evaluation.metrics import Evaluator


class RailGuardFSL:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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

    def initialize(self, healthy_embeddings: np.ndarray, threshold_percentile: float = 95.0):
        self.anomaly_detector.fit(healthy_embeddings, threshold_percentile)
        self.initialized = True

    def process_frame(self, image: np.ndarray, track_id: str = "unknown", location_m: float = 0.0) -> Dict:
        validation = self.validator.validate(image)
        if validation["status"] != "valid":
            return validation

        rail_region = self.extractor.extract_rail_region(image)
        if rail_region is None:
            return {"status": "error", "reason": "rail_extraction_failed"}

        patches = self.patcher.extract_patches(rail_region)
        patch_images = [p for p, _ in patches]
        patch_coords = [c for _, c in patches]

        embeddings = self.encoder.embed_patches(patch_images)

        if not self.initialized:
            return {"status": "not_initialized", "patches_extracted": len(patches)}

        anomaly_result = self.anomaly_detector.predict(embeddings, patch_coords, rail_region.shape[:2])

        if not anomaly_result["is_anomaly"]:
            heatmap = self.explainability.generate_anomaly_heatmap(
                rail_region, anomaly_result["anomaly_map"]
            )
            return {
                "status": "healthy",
                "anomaly_score": anomaly_result["anomaly_score"],
                "heatmap": heatmap,
            }

        if self.few_shot_classifier is not None and self.few_shot_classifier.prototypes is not None:
            open_set_results = self.few_shot_classifier.open_set_classify(
                embeddings, self.open_set_recognizer.distance_threshold
            )
            primary_result = max(open_set_results, key=lambda x: x["confidence"])
            label = primary_result["label"]
            confidence = primary_result["confidence"]

            if label != "unknown_anomaly":
                hard_neg = self.hard_negative_mgr.has_hard_negative(embeddings.mean(axis=0))
                if hard_neg:
                    return {
                        "status": "healthy",
                        "anomaly_score": anomaly_result["anomaly_score"],
                        "note": "hard_negative_suppressed",
                    }

            severity = self.severity_estimator.estimate(
                anomaly_result["anomaly_score"],
                anomaly_result["anomaly_mask"],
                confidence,
            )

            twin_result = self.twin_mgr.report_defect(
                track_id=track_id,
                location_m=location_m,
                defect_type=label,
                severity=severity["severity"],
                anomaly_score=anomaly_result["anomaly_score"],
                confidence=confidence,
            )

            heatmap = self.explainability.generate_anomaly_heatmap(
                rail_region, anomaly_result["anomaly_map"]
            )
            grad_cam = self.explainability.generate_gradcam(
                self.encoder.model, rail_region
            ) if hasattr(self.encoder.model, "get_activations_gradient") else heatmap

            return {
                "status": "defect_detected",
                "label": label,
                "confidence": confidence,
                "anomaly_score": anomaly_result["anomaly_score"],
                "severity": severity,
                "patch_coords": patch_coords,
                "failure_risk": twin_result["failure_risk_pct"],
                "event_id": twin_result["event_id"],
                "priority_rank": twin_result["priority_rankings"][0]["priority_rank"] if twin_result["priority_rankings"] else "N/A",
                "heatmap": heatmap,
                "gradcam": grad_cam,
                "anomaly_mask": anomaly_result["anomaly_mask"],
            }
        else:
            severity = self.severity_estimator.estimate(
                anomaly_result["anomaly_score"],
                anomaly_result["anomaly_mask"],
                0.5,
            )
            heatmap = self.explainability.generate_anomaly_heatmap(
                rail_region, anomaly_result["anomaly_map"]
            )
            return {
                "status": "anomaly_detected_unclassified",
                "anomaly_score": anomaly_result["anomaly_score"],
                "severity": severity,
                "heatmap": heatmap,
                "anomaly_mask": anomaly_result["anomaly_mask"],
            }

    def setup_few_shot(self, support_embeddings: Dict[str, np.ndarray], n_shots: int = 5):
        all_embs = []
        all_labels = []
        class_names = []
        for label_idx, (class_name, embs) in enumerate(support_embeddings.items()):
            if len(embs) >= n_shots:
                all_embs.append(embs[:n_shots])
                all_labels.extend([label_idx] * n_shots)
                class_names.append(class_name)
        if not all_embs:
            return False
        all_embs = np.concatenate(all_embs, axis=0)
        all_labels = np.array(all_labels)
        self.few_shot_classifier = PrototypicalNetwork(
            embedding_dim=all_embs.shape[1],
            n_ways=len(class_names),
            n_shots=n_shots,
        )
        self.few_shot_classifier.compute_prototypes(all_embs, all_labels, class_names)
        return True

    def evaluate_few_shot_performance(self, all_embeddings: Dict[str, np.ndarray], n_ways: int = 5, n_shots: int = 5, n_episodes: int = 100):
        return evaluate_few_shot(all_embeddings, n_ways, n_shots, n_episodes)


import torch
