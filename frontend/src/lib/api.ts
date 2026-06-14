import type { HealthStatus, DetectionResult, FewShotResult, Stats, Segment, UnknownSample, DefectDistribution, ConformalResult, DomainCalibration, AugmentResult } from "./types"

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json()
}

export const api = {
  health: () => request<HealthStatus>("/health"),

  initialize: (threshold_percentile = 95) =>
    request<{ status: string; message: string }>("/initialize", {
      method: "POST",
      body: JSON.stringify({ threshold_percentile }),
    }),

  setupFewShot: (classes: string[], n_shots = 5) => {
    const params = new URLSearchParams()
    classes.forEach((c) => params.append("classes", c))
    params.append("n_shots", String(n_shots))
    return fetch(`${API}/few-shot/setup`, { method: "POST", body: params }).then((r) => r.json())
  },

  detect: (file: File, track_id?: string, location_m?: number) => {
    const TRACKS = ["Track_A12", "Track_B7", "Track_C3", "Track_D9", "Track_E5"]
    const tid = track_id ?? TRACKS[Math.floor(Math.random() * TRACKS.length)]
    const loc = location_m ?? Math.floor(Math.random() * 900)
    const form = new FormData()
    form.append("file", file)
    form.append("track_id", tid)
    form.append("location_m", String(loc))
    return fetch(`${API}/detect`, { method: "POST", body: form }).then((r) => r.json()) as Promise<DetectionResult>
  },

  evaluateFewShot: (n_ways: number, n_shots: number, n_episodes: number) =>
    request<FewShotResult>("/evaluate/few-shot", {
      method: "POST",
      body: JSON.stringify({ n_ways, n_shots, n_episodes }),
    }),

  stats: () => request<Stats>("/stats"),

  defectDistribution: () => request<DefectDistribution>("/stats/defect-distribution"),

  twinStatus: (track_id: string) => request<Segment>(`/digital-twin/status/${track_id}`),

  twinOverview: () => request<{ segments: Segment[] }>("/digital-twin/status"),

  twinReport: (track_id: string) =>
    request<Segment>("/digital-twin/report", {
      method: "POST",
      body: JSON.stringify({ track_id, location_m: 0 }),
    }),

  unknownSamples: () => request<UnknownSample[]>("/discovery/unknowns"),

  labelUnknown: (id: string, label: string) =>
    request<{ status: string; message: string }>("/discovery/label", {
      method: "POST",
      body: JSON.stringify({ id, label }),
    }),

  conformalCalibrate: (target_recall: number) =>
    request<ConformalResult>("/conformal/calibrate", {
      method: "POST",
      body: JSON.stringify({ target_recall }),
    }),

  conformalStatus: () =>
    request<{ conformal: ConformalResult | null; scores: { n_defect: number; n_healthy: number } }>("/conformal/status"),

  calibrateThreshold: (percentile: number, domain = "default") =>
    request<DomainCalibration>("/calibrate/threshold", {
      method: "POST",
      body: JSON.stringify({ percentile, domain }),
    }),

  calibrateDomains: () => request<{ domains: DomainCalibration[] }>("/calibrate/domains"),

  augmentPreview: (file: File, kind: string) => {
    const form = new FormData()
    form.append("file", file)
    form.append("kind", kind)
    return fetch(`${API}/augment/preview`, { method: "POST", body: form }).then((r) => r.json()) as Promise<AugmentResult>
  },
}
