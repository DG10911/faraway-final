"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import {
  Activity, AlertTriangle, Clock, Cpu, Zap, AlertCircle, Info,
  Play, Pause, Video, Radio, MapPin, ScanLine, CheckCircle2, HelpCircle,
} from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { StatusDot } from "@/components/ui/StatusDot"
import { api } from "@/lib/api"
import type { Stats, DetectionResult } from "@/lib/types"

type Mode = "track" | "webcam"
interface DemoItem { src: string; label: string }
interface Live { preview: string; result: DetectionResult; expected: string; track: string; loc: number }

const FRAME_INTERVAL_MS = 1900
const TRACKS = ["Track_A12", "Track_B7", "Track_C3", "Track_D9", "Track_E5"]

const FALLBACK: DemoItem[] = [
  ...[1, 2, 3].map((n) => ({ src: `/demo/healthy_${n}.jpg`, label: "healthy" })),
  ...[1, 2, 3].map((n) => ({ src: `/demo/crack_${n}.jpg`, label: "crack" })),
  ...[1, 2].map((n) => ({ src: `/demo/squat_${n}.jpg`, label: "squat" })),
  ...[1, 2].map((n) => ({ src: `/demo/spalling_${n}.jpg`, label: "spalling" })),
  ...[1, 2].map((n) => ({ src: `/demo/shelling_${n}.jpg`, label: "shelling" })),
  ...[1, 2].map((n) => ({ src: `/demo/groove_${n}.jpg`, label: "groove" })),
  ...[1, 2].map((n) => ({ src: `/demo/joint_${n}.jpg`, label: "joint" })),
  ...[1, 2].map((n) => ({ src: `/demo/flaking_${n}.jpg`, label: "flaking" })),
]

const SEVERITY_VARIANT: Record<string, "error" | "warning" | "success" | "info"> = {
  Critical: "error", High: "warning", Medium: "warning", Low: "success",
}

// classify the result into a visual tone for the feed overlay
function tone(result?: DetectionResult) {
  const s = result?.status
  if (s === "defect_detected") return { key: "defect", label: "DEFECT", color: "text-danger-400", ring: "ring-danger-500/60", bar: "bg-danger-500", Icon: AlertTriangle, glow: "anomaly" as const }
  if (s === "healthy" || s === "valid") return { key: "healthy", label: "HEALTHY", color: "text-emerald-400", ring: "ring-emerald-500/50", bar: "bg-emerald-500", Icon: CheckCircle2, glow: "healthy" as const }
  if (s === "unknown_anomaly" || s === "anomaly_detected_unclassified") return { key: "unknown", label: "UNKNOWN", color: "text-amber-400", ring: "ring-amber-500/50", bar: "bg-amber-500", Icon: HelpCircle, glow: "none" as const }
  return { key: "rejected", label: "NO RAIL", color: "text-slate-400", ring: "ring-slate-600/50", bar: "bg-slate-500", Icon: Info, glow: "none" as const }
}

export default function LiveMonitoring() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [offline, setOffline] = useState(false)

  const [items, setItems] = useState<DemoItem[]>([])
  const [mode, setMode] = useState<Mode>("track")
  const [playing, setPlaying] = useState(true)
  const [live, setLive] = useState<Live | null>(null)
  const [detecting, setDetecting] = useState(false)
  const [camError, setCamError] = useState<string | null>(null)

  const idxRef = useRef(0)
  const busyRef = useRef(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // ---- poll cumulative pipeline stats (drives the metric cards) ----
  useEffect(() => {
    const controller = new AbortController()
    const load = async () => {
      try {
        const s = await api.stats()
        if (!controller.signal.aborted) { setStats(s); setOffline(false) }
      } catch {
        if (!controller.signal.aborted) setOffline(true)
      }
    }
    load()
    const id = setInterval(load, 2000)
    return () => { controller.abort(); clearInterval(id) }
  }, [])

  // ---- load the demo gallery manifest (the "track camera" frames) ----
  useEffect(() => {
    fetch("/demo/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setItems(Array.isArray(d) && d.length ? d : FALLBACK))
      .catch(() => setItems(FALLBACK))
  }, [])

  // ---- webcam lifecycle: open the camera in webcam mode, release otherwise ----
  useEffect(() => {
    if (mode !== "webcam") {
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      return
    }
    let cancelled = false
    navigator.mediaDevices
      ?.getUserMedia({ video: { facingMode: "environment" } })
      .then((stream) => {
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return }
        streamRef.current = stream
        setCamError(null)
        if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play().catch(() => {}) }
      })
      .catch(() => setCamError("Camera unavailable — allow access, or switch to Track Camera."))
    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [mode])

  // ---- grab one frame as a File (+ preview) from the active source ----
  const grabFrame = useCallback(async (): Promise<{ file: File; preview: string; expected: string } | null> => {
    if (mode === "track") {
      if (!items.length) return null
      const item = items[idxRef.current % items.length]
      idxRef.current += 1
      const blob = await (await fetch(item.src)).blob()
      const file = new File([blob], item.src.split("/").pop() || "frame.jpg", { type: blob.type || "image/jpeg" })
      return { file, preview: item.src, expected: item.label }
    }
    const v = videoRef.current, c = canvasRef.current
    if (!v || !c || !v.videoWidth) return null
    c.width = v.videoWidth; c.height = v.videoHeight
    const ctx = c.getContext("2d")
    if (!ctx) return null
    ctx.drawImage(v, 0, 0, c.width, c.height)
    const preview = c.toDataURL("image/jpeg", 0.8)
    const blob: Blob | null = await new Promise((res) => c.toBlob(res, "image/jpeg", 0.8))
    if (!blob) return null
    return { file: new File([blob], "webcam.jpg", { type: "image/jpeg" }), preview, expected: "live" }
  }, [mode, items])

  // ---- the live loop: capture → /detect → render, on a fixed cadence ----
  useEffect(() => {
    if (!playing) return
    if (mode === "track" && !items.length) return
    let stop = false
    const tick = async () => {
      if (stop || busyRef.current) return
      busyRef.current = true
      setDetecting(true)
      try {
        const grabbed = await grabFrame()
        if (grabbed && !stop) {
          const track = TRACKS[Math.floor(Math.random() * TRACKS.length)]
          const loc = Math.floor(Math.random() * 900)
          const result = await api.detect(grabbed.file, track, loc)
          if (!stop) setLive({ preview: grabbed.preview, result, expected: grabbed.expected, track, loc })
        }
      } catch {
        /* offline state is surfaced by the stats poll */
      } finally {
        busyRef.current = false
        if (!stop) setDetecting(false)
      }
    }
    tick()
    const id = setInterval(tick, FRAME_INTERVAL_MS)
    return () => { stop = true; clearInterval(id) }
  }, [playing, mode, items, grabFrame])

  const t = tone(live?.result)
  const result = live?.result
  const conf = result?.confidence
  const metrics = stats
    ? [
        { label: "Frames Processed", value: stats.frames_processed.toLocaleString(), icon: Cpu },
        { label: "Frames Rejected", value: stats.frames_rejected.toLocaleString(), icon: AlertCircle },
        { label: "Anomalies Found", value: stats.anomalies_found, icon: Activity, danger: stats.anomalies_found > 0 },
        { label: "Defects Confirmed", value: stats.defects_confirmed, icon: AlertTriangle, danger: stats.defects_confirmed > 0 },
        { label: "Unknowns Flagged", value: stats.unknowns_flagged, icon: Info, warn: stats.unknowns_flagged > 0 },
        { label: "Avg Latency", value: stats.avg_latency_ms ? `${stats.avg_latency_ms.toFixed(0)}ms` : "—", icon: Clock },
      ]
    : []

  return (
    <PageContainer
      title="Live Monitoring"
      subtitle="Streaming rail inspection — each frame runs through the full DINOv2 → PatchCore → few-shot pipeline."
      actions={<StatusDot status={offline ? "offline" : playing ? "online" : "warning"} label={offline ? "Disconnected" : playing ? "Streaming" : "Paused"} />}
    >
      {offline && (
        <div className="flex items-center gap-3 p-4 mb-6 rounded-lg bg-red-500/15 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={18} className="shrink-0" />
          <span>API is offline — start the backend (python run_api.py). Reconnecting…</span>
        </div>
      )}

      {/* controls */}
      <div className="flex flex-wrap items-center gap-3 mb-5">
        <div className="inline-flex rounded-xl border border-slate-700 bg-slate-800/50 p-1">
          {(["track", "webcam"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => { setMode(m); setLive(null) }}
              className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition-all ${mode === m ? "bg-brand-gradient text-white" : "text-slate-400 hover:text-white"}`}
            >
              {m === "track" ? <ScanLine size={14} /> : <Video size={14} />}
              {m === "track" ? "Track Camera" : "Webcam"}
            </button>
          ))}
        </div>
        <button onClick={() => setPlaying((p) => !p)} className={`btn ${playing ? "btn-ghost" : "btn-primary"}`}>
          {playing ? <><Pause size={15} /> Pause</> : <><Play size={15} /> Resume</>}
        </button>
        <span className="text-xs text-slate-500">
          {mode === "track" ? "Cycling demo track frames through the live pipeline" : "Analyzing your webcam feed frame-by-frame"}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* ---- camera feed ---- */}
        <div className="lg:col-span-2">
          <div className={`relative aspect-video w-full overflow-hidden rounded-2xl border border-slate-700 bg-black ring-1 ${t.ring} transition-all`}>
            {/* webcam stream stays mounted so the ref is stable; covered by the img in track mode */}
            <video ref={videoRef} autoPlay muted playsInline className="absolute inset-0 h-full w-full object-cover" />
            {mode === "track" && live && (
              <img src={live.preview} alt="feed" className="absolute inset-0 h-full w-full object-cover" />
            )}
            <canvas ref={canvasRef} className="hidden" />

            {/* sweeping scan line */}
            {playing && <div className="pointer-events-none absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-transparent via-rail-300 to-transparent animate-scanline shadow-[0_0_12px_2px_rgba(124,135,255,0.6)]" />}

            {/* camera framing brackets */}
            <div className="pointer-events-none absolute inset-3">
              {["top-0 left-0 border-t-2 border-l-2", "top-0 right-0 border-t-2 border-r-2", "bottom-0 left-0 border-b-2 border-l-2", "bottom-0 right-0 border-b-2 border-r-2"].map((c, i) => (
                <span key={i} className={`absolute h-5 w-5 ${c} border-white/30 rounded-sm`} />
              ))}
            </div>

            {/* top bar: LIVE + source */}
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between">
              <span className="inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11px] font-bold tracking-wider text-red-400 backdrop-blur">
                <span className="h-2 w-2 rounded-full bg-red-500 animate-live-pulse" />
                {playing ? "LIVE" : "PAUSED"}
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-md bg-black/60 px-2 py-1 text-[11px] font-mono text-slate-300 backdrop-blur">
                {mode === "track" ? <><MapPin size={11} className="text-rail-300" />{live ? `${live.track} · ${live.loc}m` : "track camera"}</> : <><Video size={11} className="text-rail-300" /> webcam</>}
              </span>
            </div>

            {/* bottom status strip */}
            <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/85 to-transparent px-4 pb-3 pt-8">
              <div className="flex items-end justify-between gap-3">
                <div className="flex items-center gap-2">
                  <t.Icon size={18} className={t.color} />
                  <div>
                    <p className={`text-sm font-bold tracking-wide ${t.color}`}>{t.label}</p>
                    {result?.label && <p className="text-xs text-slate-300 capitalize">{result.label.replace(/_/g, " ")}{conf !== undefined ? ` · ${(conf * 100).toFixed(0)}%` : ""}</p>}
                  </div>
                </div>
                {detecting && <span className="flex items-center gap-1.5 text-[11px] text-rail-300"><Radio size={12} className="animate-pulse" /> analyzing…</span>}
              </div>
            </div>

            {/* empty / error states */}
            {!live && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-slate-400">
                {camError ? (
                  <><AlertCircle size={28} className="mb-2 text-amber-400" /><p className="text-sm max-w-xs px-4">{camError}</p></>
                ) : (
                  <><ScanLine size={28} className="mb-2 animate-pulse text-rail-300" /><p className="text-sm">Acquiring feed…</p></>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ---- live detection readout ---- */}
        <Card glow={t.glow} className="flex flex-col">
          <p className="eyebrow mb-3"><Activity size={13} /> Current frame</p>
          {result ? (
            <div className="space-y-4 animate-fade-up">
              <div className="flex items-center gap-2">
                <t.Icon size={20} className={t.color} />
                <span className={`text-lg font-bold ${t.color}`}>{(result.status || "").replace(/_/g, " ")}</span>
              </div>

              {result.label && (
                <div className="flex items-center justify-between rounded-xl bg-rail-500/15 border border-rail-500/20 px-3.5 py-2.5">
                  <span className="text-sm text-slate-400">Defect type</span>
                  <span className="font-bold capitalize text-rail-300">{result.label.replace(/_/g, " ")}</span>
                </div>
              )}

              {mode === "track" && live && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-500">Ground truth</span>
                  <span className="font-mono capitalize text-slate-400">{live.expected}</span>
                </div>
              )}

              {conf !== undefined && (
                <div>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="text-slate-400">Confidence</span>
                    <span className="font-mono text-slate-100">{(conf * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-brand-gradient rounded-full transition-all duration-500" style={{ width: `${(conf * 100).toFixed(0)}%` }} />
                  </div>
                </div>
              )}

              {result.anomaly_score !== undefined && (
                <div className="flex items-center justify-between text-sm">
                  <span className="text-slate-400">Anomaly score</span>
                  <span className="font-mono text-cyan-400">{result.anomaly_score.toFixed(4)}</span>
                </div>
              )}

              {result.severity && (
                <div className="flex items-center gap-2">
                  <span className="text-sm text-slate-400">Severity</span>
                  <Badge variant={SEVERITY_VARIANT[result.severity.severity] || "default"}>{result.severity.severity}</Badge>
                </div>
              )}

              {result.heatmap_b64 && (
                <div>
                  <p className="eyebrow mb-1.5">Anomaly heatmap</p>
                  <img src={`data:image/png;base64,${result.heatmap_b64}`} alt="heatmap" className="w-full rounded-lg border border-slate-700" />
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center text-center text-slate-500 min-h-[200px]">
              <Cpu size={30} className="mb-2 text-slate-600" />
              <p className="text-sm">Waiting for the first frame…</p>
            </div>
          )}
        </Card>
      </div>

      {/* ---- cumulative pipeline metrics ---- */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-6">
          {metrics.map((m) => {
            const Icon = m.icon
            return (
              <Card key={m.label} className="text-center" glow={m.danger ? "anomaly" : "none"}>
                <Icon size={20} className={`mx-auto mb-2 ${m.danger ? "text-danger-400" : m.warn ? "text-amber-400" : "text-rail-300"}`} />
                <p className="text-2xl font-bold font-mono">{m.value}</p>
                <p className="text-xs text-slate-400 mt-1">{m.label}</p>
              </Card>
            )
          })}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Zap size={18} className="text-rail-300" />
            <h3 className="font-semibold">Throughput</h3>
          </div>
          {stats?.throughput_fps != null ? (
            <div className="flex items-baseline gap-2">
              <span className="text-4xl font-bold font-mono text-emerald-400">{stats.throughput_fps.toFixed(1)}</span>
              <span className="text-slate-400">fps</span>
            </div>
          ) : (
            <p className="text-sm text-slate-400">Waiting for data…</p>
          )}
          <div className="mt-4 h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-green-500 rounded-full transition-all duration-1000" style={{ width: `${Math.min(((stats?.throughput_fps ?? 0) / 60) * 100, 100)}%` }} />
          </div>
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-4">
            <AlertTriangle size={18} className="text-rail-300" />
            <h3 className="font-semibold">Recent Alerts</h3>
          </div>
          {(stats?.alerts ?? []).length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {stats!.alerts.map((alert, i) => (
                <div key={i} className="flex items-start gap-3 p-2 bg-slate-800/50 rounded-lg text-sm">
                  <span className="w-2 h-2 rounded-full bg-danger-500 mt-1.5 shrink-0" />
                  <div>
                    <p>{alert.message}</p>
                    <p className="text-xs text-slate-400 mt-0.5">{alert.timestamp}</p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-400">No alerts yet</p>
          )}
        </Card>
      </div>
    </PageContainer>
  )
}
