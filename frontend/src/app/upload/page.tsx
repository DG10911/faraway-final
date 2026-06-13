"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Upload, Image as ImageIcon, AlertTriangle, CheckCircle, XCircle, Loader2 } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { api } from "@/lib/api"
import type { DetectionResult } from "@/lib/types"

const SEVERITY_VARIANT: Record<string, "error" | "warning" | "success" | "info"> = {
  Critical: "error",
  High: "warning",
  Medium: "warning",
  Low: "success",
}

export default function UploadDetection() {
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0]
    if (!file) return

    setPreview(URL.createObjectURL(file))
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

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg"] },
    maxFiles: 1,
  })

  return (
    <PageContainer title="Upload Detection" subtitle="Analyze a single rail image for defects">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div
          {...getRootProps()}
          className={`glass rounded-xl p-12 text-center cursor-pointer border-2 border-dashed transition-all flex flex-col items-center justify-center min-h-[320px] ${
            isDragActive ? "border-rail-400 bg-rail-900/20" : "border-gray-700 hover:border-gray-500"
          }`}
        >
          <input {...getInputProps()} />
          <div className={`w-16 h-16 rounded-full flex items-center justify-center mb-4 ${isDragActive ? "bg-rail-900/60" : "bg-gray-800"}`}>
            <Upload size={28} className={isDragActive ? "text-rail-400" : "text-gray-500"} />
          </div>
          {isDragActive ? (
            <p className="text-rail-400 font-medium">Drop image here...</p>
          ) : (
            <>
              <p className="text-gray-300 font-medium">Drag & drop a rail image</p>
              <p className="text-xs text-gray-500 mt-1">or click to select — PNG, JPG up to 16MB</p>
            </>
          )}
        </div>

        <Card>
          {loading ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[320px]">
              <Loader2 size={32} className="text-rail-400 animate-spin mb-3" />
              <p className="text-gray-400">Analyzing frame...</p>
            </div>
          ) : result ? (
            <div className="space-y-5">
              <div className="flex items-center gap-3">
                {result.status === "defect_detected" ? (
                  <div className="w-10 h-10 rounded-full bg-red-900/40 flex items-center justify-center">
                    <AlertTriangle size={20} className="text-danger-400" />
                  </div>
                ) : result.status === "healthy" || result.status === "valid" ? (
                  <div className="w-10 h-10 rounded-full bg-green-900/40 flex items-center justify-center">
                    <CheckCircle size={20} className="text-green-400" />
                  </div>
                ) : (
                  <div className="w-10 h-10 rounded-full bg-yellow-900/40 flex items-center justify-center">
                    <XCircle size={20} className="text-yellow-400" />
                  </div>
                )}
                <div>
                  <p className="font-semibold capitalize">{result.status?.replace("_", " ")}</p>
                  {result.reason && (
                    <p className="text-sm text-gray-400">{result.reason.replace("_", " ")}</p>
                  )}
                </div>
              </div>

              {result.label && (
                <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                  <span className="text-sm text-gray-400">Defect Type</span>
                  <span className="font-bold">{result.label.replace("_", " ")}</span>
                </div>
              )}

              {result.confidence !== undefined && (
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-400">Confidence</span>
                    <span className="font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-rail-400 rounded-full transition-all"
                      style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                    />
                  </div>
                </div>
              )}

              {result.anomaly_score !== undefined && (
                <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                  <span className="text-sm text-gray-400">Anomaly Score</span>
                  <span className="font-mono text-danger-400">{result.anomaly_score.toFixed(4)}</span>
                </div>
              )}

              {result.severity && (
                <div className="flex items-center gap-3">
                  <span className="text-sm text-gray-400">Severity</span>
                  <Badge variant={SEVERITY_VARIANT[result.severity.severity] || "default"}>{result.severity.severity}</Badge>
                </div>
              )}

              {result.failure_risk !== undefined && (
                <div className="flex items-center justify-between p-3 bg-gray-900 rounded-lg">
                  <span className="text-sm text-gray-400">Failure Risk</span>
                  <span className="font-mono text-danger-400">{result.failure_risk}%</span>
                </div>
              )}

              {result.heatmap_b64 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Anomaly Heatmap</p>
                  <img
                    src={`data:image/png;base64,${result.heatmap_b64}`}
                    alt="Heatmap"
                    className="w-full rounded-lg border border-gray-800"
                  />
                </div>
              )}

              {result.gradcam_b64 && (
                <div>
                  <p className="text-sm text-gray-400 mb-2">Grad-CAM</p>
                  <img
                    src={`data:image/png;base64,${result.gradcam_b64}`}
                    alt="Grad-CAM"
                    className="w-full rounded-lg border border-gray-800"
                  />
                </div>
              )}
            </div>
          ) : preview ? (
            <div className="flex flex-col items-center justify-center h-full min-h-[320px]">
              <Loader2 size={24} className="text-gray-500 animate-spin mb-2" />
              <p className="text-gray-500">Awaiting analysis...</p>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center h-full min-h-[320px]">
              <ImageIcon size={40} className="text-gray-700 mb-3" />
              <p className="text-gray-500">Upload an image to begin</p>
            </div>
          )}
        </Card>
      </div>
    </PageContainer>
  )
}
