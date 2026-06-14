"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, Image as ImageIcon, AlertTriangle, CheckCircle, XCircle, Loader2, Gauge, Activity, ShieldAlert } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { DemoGallery } from "@/components/ui/DemoGallery"
import { api } from "@/lib/api"
import type { DetectionResult } from "@/lib/types"

const SEVERITY_VARIANT: Record<string, "error" | "warning" | "success" | "info"> = {
  Critical: "error", High: "warning", Medium: "warning", Low: "success",
}

function Metric({ icon: Icon, label, value, tone = "text-slate-100" }: any) {
  return (
    <div className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-800/50 px-3.5 py-3">
      <span className="flex items-center gap-2 text-sm text-slate-400"><Icon size={15} className="text-slate-500" />{label}</span>
      <span className={`font-mono text-sm font-semibold ${tone}`}>{value}</span>
    </div>
  )
}

export default function UploadDetection() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  const analyze = useCallback(async (file: File, previewUrl: string) => {
    setPreview(previewUrl)
    setLoading(true)
    setResult(null)
    try {
      const data = await api.detect(file)
      setResult(data)
    } catch {
      setResult({ status: "error", reason: "API unreachable" })
    } finally {
      setLoading(false)
    }
  }, [])

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return
    analyze(file, URL.createObjectURL(file))
  }, [analyze])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "image/*": [".png", ".jpg", ".jpeg"] }, maxFiles: 1,
  })

  const isDefect = result?.status === "defect_detected"
  const isHealthy = result?.status === "healthy" || result?.status === "valid"
  const isUnknown = result?.status === "unknown_anomaly" || result?.status === "anomaly_detected_unclassified"

  return (
    <PageContainer
      eyebrow="Detect"
      title="Upload Detection"
      subtitle="Drop a rail image and watch the two-stage pipeline screen, classify, and explain it in real time."
    >
      <div className="mb-6">
        <DemoGallery onPick={(file, preview) => analyze(file, preview)} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* LEFT: dropzone + preview */}
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={`relative overflow-hidden rounded-2xl p-10 text-center cursor-pointer border-2 border-dashed transition-all flex flex-col items-center justify-center min-h-[260px] ${
              isDragActive ? "border-rail-400 bg-rail-500/15" : "border-slate-600 hover:border-rail-400 bg-slate-800/50"
            }`}
          >
            <input {...getInputProps()} />
            <div className={`w-16 h-16 rounded-2xl flex items-center justify-center mb-4 transition-all ${isDragActive ? "bg-brand-gradient scale-110" : "bg-brand-soft border border-rail-500/20"}`}>
              <Upload size={26} className={isDragActive ? "text-white" : "text-rail-300"} />
            </div>
            {isDragActive ? (
              <p className="text-rail-300 font-semibold">Drop to analyze…</p>
            ) : (
              <>
                <p className="text-slate-100 font-semibold">Drag &amp; drop a rail image</p>
                <p className="text-xs text-slate-400 mt-1">or click to browse — PNG / JPG up to 16 MB</p>
              </>
            )}
          </div>

          {preview && (
            <Card className="p-3">
              <p className="eyebrow mb-2 px-1">Input frame</p>
              <img src={preview} alt="Preview" className="w-full rounded-xl border border-slate-700" />
            </Card>
          )}
        </div>

        {/* RIGHT: result */}
        <Card glow={isDefect ? "anomaly" : isHealthy ? "healthy" : "none"} className="min-h-[260px]">
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[300px]">
              <Loader2 size={34} className="text-rail-300 animate-spin mb-3" />
              <p className="text-slate-400">Analyzing frame…</p>
              <p className="text-xs text-slate-500 mt-1">DINOv2 → PatchCore → prototypes</p>
            </div>
          ) : result ? (
            <div className="space-y-5 animate-fade-up">
              <div className="flex items-center gap-3">
                {isDefect ? (
                  <div className="w-11 h-11 rounded-xl bg-red-500/15 border border-red-500/30 flex items-center justify-center"><AlertTriangle size={20} className="text-danger-400" /></div>
                ) : isHealthy ? (
                  <div className="w-11 h-11 rounded-xl bg-emerald-500/15 border border-emerald-500/30 flex items-center justify-center"><CheckCircle size={20} className="text-emerald-400" /></div>
                ) : (
                  <div className="w-11 h-11 rounded-xl bg-amber-500/15 border border-amber-500/30 flex items-center justify-center"><XCircle size={20} className="text-amber-400" /></div>
                )}
                <div>
                  <p className="font-bold text-lg capitalize leading-tight text-slate-100">{result.status?.replace(/_/g, " ")}</p>
                  {result.reason && <p className="text-sm text-slate-400">{result.reason.replace(/_/g, " ")}</p>}
                  {isUnknown && <p className="text-sm text-amber-400">Novel pattern — send to Discovery to label it.</p>}
                </div>
              </div>

              {result.label && (
                <div className="flex items-center justify-between rounded-xl bg-rail-500/15 border border-rail-500/20 px-4 py-3">
                  <span className="text-sm text-slate-400">Defect type</span>
                  <span className="font-bold capitalize text-rail-300">{result.label.replace(/_/g, " ")}</span>
                </div>
              )}

              {result.confidence !== undefined && (
                <div>
                  <div className="flex justify-between text-sm mb-1.5">
                    <span className="text-slate-400">Confidence</span>
                    <span className="font-mono text-slate-100">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2.5 bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full bg-brand-gradient rounded-full transition-all duration-700" style={{ width: `${(result.confidence * 100).toFixed(0)}%` }} />
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 gap-2.5">
                {result.anomaly_score !== undefined && (
                  <Metric icon={Activity} label="Anomaly score" value={result.anomaly_score.toFixed(4)} tone="text-cyan-400" />
                )}
                {result.failure_risk !== undefined && (
                  <Metric icon={ShieldAlert} label="Failure risk" value={`${result.failure_risk}%`} tone="text-danger-400" />
                )}
                {result.priority_rank && result.priority_rank !== "N/A" && (
                  <Metric icon={Gauge} label="Maintenance priority" value={`#${result.priority_rank}`} tone="text-amber-400" />
                )}
              </div>

              {result.severity && (
                <div className="flex items-center gap-3">
                  <span className="text-sm text-slate-400">Severity</span>
                  <Badge variant={SEVERITY_VARIANT[result.severity.severity] || "default"}>{result.severity.severity}</Badge>
                </div>
              )}

              {result.heatmap_b64 && (
                <div>
                  <p className="eyebrow mb-2">Anomaly heatmap</p>
                  <img src={`data:image/png;base64,${result.heatmap_b64}`} alt="Heatmap" className="w-full rounded-xl border border-slate-700" />
                </div>
              )}
              {(result.attention_b64 || result.gradcam_b64) && (
                <div>
                  <p className="eyebrow mb-2">DINOv2 attention</p>
                  <img src={`data:image/png;base64,${result.attention_b64 || result.gradcam_b64}`} alt="Attention" className="w-full rounded-xl border border-slate-700" />
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center">
              <ImageIcon size={42} className="text-slate-600 mb-3" />
              <p className="text-slate-400">Results will appear here</p>
              <p className="text-xs text-slate-500 mt-1">Upload an image to begin</p>
            </div>
          )}
        </Card>
      </div>
    </PageContainer>
  )
}
