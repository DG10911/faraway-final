export interface HealthStatus {
  status: string
  model: string
  initialized: boolean
}

export interface DetectionResult {
  status: string
  reason?: string
  label?: string
  confidence?: number
  anomaly_score?: number
  severity?: SeverityResult
  failure_risk?: number
  event_id?: string
  priority_rank?: string
  heatmap_b64?: string
  attention_b64?: string
  gradcam_b64?: string
  anomaly_mask_b64?: string
  patch_coords?: number[][]
  blur_score?: number
  quality_score?: number
  occlusion_score?: number
}

export interface SeverityResult {
  severity: "Low" | "Medium" | "High" | "Critical"
  defect_area: number
  anomaly_score: number
  confidence: number
  defect_density: number
  composite_score: number
}

export interface FewShotResult {
  mean_accuracy: number
  std_accuracy: number
  n_episodes: number
  open_set_detection_rate?: number
  ci_95?: number
  error?: string
}

export interface Stats {
  frames_processed: number
  frames_rejected: number
  anomalies_found: number
  defects_confirmed: number
  unknowns_flagged: number
  avg_latency_ms: number | null
  throughput_fps: number | null
  initialized: boolean
  few_shot_ready: boolean
  alerts: Alert[]
}

export interface Alert {
  message: string
  timestamp: string
}

export interface Ranking {
  event_id: string
  track_id: string
  defect_type: string
  severity: string
  failure_risk_pct: number
  location_m: number
  priority_rank: string
  confidence?: number
}

export interface Segment {
  track_id: string
  active_defects: number
  priority_rankings: Ranking[]
  overall_health: number
  total_length_m?: number
  status?: string
}

export interface UnknownSample {
  id: string
  timestamp: string
  track_id: string
  anomaly_score: number
  distance: number | null
  thumbnail_b64: string | null
  confidence?: number
  label?: string
}

export interface DefectDistribution {
  distribution: Record<string, number>
}

export interface ConformalResult {
  threshold: number
  target_recall: number
  guaranteed_recall: number
  n_calibration: number
  guarantee_achievable: boolean
  note: string
  empirical_recall?: number | null
  false_positive_rate?: number | null
  n_defect?: number
  n_healthy?: number
  applied?: boolean
  error?: string
}

export interface DomainCalibration {
  domain: string
  percentile: number
  threshold: number
  false_positive_rate?: number | null
  empirical_recall?: number | null
  n_defect?: number
}

export interface AugmentResult {
  kind: string
  original_b64?: string
  synthetic_b64?: string
}

export type ViewKey = "heatmap" | "attention" | "mask"

export const SEVERITY_COLORS: Record<string, string> = {
  Low: "text-emerald-300 bg-emerald-500/15 border-emerald-500/30",
  Medium: "text-amber-400 bg-amber-500/15 border-amber-500/30",
  High: "text-orange-400 bg-orange-500/15 border-orange-500/30",
  Critical: "text-red-400 bg-red-500/15 border-red-500/30",
}
