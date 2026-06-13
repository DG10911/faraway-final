import numpy as np
import json
import uuid
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class DefectEvent:
    track_id: str
    location_m: float
    defect_type: str
    severity: str
    anomaly_score: float
    confidence: float
    timestamp: str = ""
    event_id: str = ""

    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class RailwaySegment:
    def __init__(self, track_id: str, total_length_m: float = 1000.0):
        self.track_id = track_id
        self.total_length = total_length_m
        self.defects: List[DefectEvent] = []
        self.maintenance_history: List[Dict] = []

    def add_defect(self, defect: DefectEvent):
        self.defects.append(defect)

    def compute_failure_risk(self, defect: DefectEvent) -> float:
        severity_weights = {"Low": 0.2, "Medium": 0.4, "High": 0.7, "Critical": 0.95}
        severity_weight = severity_weights.get(defect.severity, 0.3)
        base_risk = defect.anomaly_score * 0.6 + severity_weight * 0.4
        risk = min(base_risk * (1 + 0.1 * len([d for d in self.defects if d.track_id == defect.track_id])), 1.0)
        return round(risk, 4)

    def get_priority_rankings(self) -> List[Dict]:
        rankings = []
        for defect in self.defects:
            risk = self.compute_failure_risk(defect)
            rankings.append({
                "event_id": defect.event_id,
                "track_id": defect.track_id,
                "defect_type": defect.defect_type,
                "severity": defect.severity,
                "failure_risk_pct": round(risk * 100, 1),
                "location_m": defect.location_m,
                "confidence": defect.confidence,
            })
        rankings.sort(key=lambda x: x["failure_risk_pct"], reverse=True)
        for i, r in enumerate(rankings):
            r["priority_rank"] = f"#{i + 1}"
        return rankings


class DigitalTwinManager:
    def __init__(self):
        self.segments: Dict[str, RailwaySegment] = {}

    def create_segment(self, track_id: str, length_m: float = 1000.0) -> RailwaySegment:
        segment = RailwaySegment(track_id, length_m)
        self.segments[track_id] = segment
        return segment

    def report_defect(
        self,
        track_id: str,
        location_m: float,
        defect_type: str,
        severity: str,
        anomaly_score: float,
        confidence: float,
    ) -> Dict:
        if track_id not in self.segments:
            self.create_segment(track_id)
        segment = self.segments[track_id]
        event = DefectEvent(
            track_id=track_id,
            location_m=location_m,
            defect_type=defect_type,
            severity=severity,
            anomaly_score=anomaly_score,
            confidence=confidence,
        )
        segment.add_defect(event)
        risk = segment.compute_failure_risk(event)
        rankings = segment.get_priority_rankings()
        return {
            "event_id": event.event_id,
            "track_id": track_id,
            "defect_type": defect_type,
            "severity": severity,
            "failure_risk_pct": round(risk * 100, 1),
            "location_m": location_m,
            "confidence": confidence,
            "priority_rankings": rankings,
            "timestamp": event.timestamp,
        }

    def get_segment_status(self, track_id: str) -> Dict:
        if track_id not in self.segments:
            return {"track_id": track_id, "status": "no_data"}
        segment = self.segments[track_id]
        return {
            "track_id": track_id,
            "total_length_m": segment.total_length,
            "active_defects": len(segment.defects),
            "priority_rankings": segment.get_priority_rankings(),
            "overall_health": self._compute_health_score(segment),
        }

    def _compute_health_score(self, segment: RailwaySegment) -> float:
        if not segment.defects:
            return 1.0
        avg_risk = np.mean([segment.compute_failure_risk(d) for d in segment.defects])
        return round(max(0.0, 1.0 - avg_risk), 3)
