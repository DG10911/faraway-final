import type { HealthStatus, DetectionResult, FewShotResult, Stats, Segment, UnknownSample, DefectDistribution } from "./types"

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

  detect: (file: File, track_id = "Upload_Track", location_m = 0) => {
    const form = new FormData()
    form.append("file", file)
    form.append("track_id", track_id)
    form.append("location_m", String(location_m))
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
}
