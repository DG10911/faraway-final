"use client"

import { useState, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { Eye, Image as ImageIcon, Upload, Loader2, Download, Thermometer, Target, Grid3X3 } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { DemoGallery } from "@/components/ui/DemoGallery"
import { api } from "@/lib/api"
import type { DetectionResult, ViewKey } from "@/lib/types"

const VIEWS: { key: ViewKey; label: string; icon: typeof Eye; desc: string }[] = [
  { key: "heatmap", label: "Anomaly Heatmap", icon: Thermometer, desc: "PatchCore anomaly scores per region" },
  { key: "attention", label: "Attention Map", icon: Target, desc: "DINOv2 self-attention visualization" },
  { key: "mask", label: "Anomaly Mask", icon: Grid3X3, desc: "Binary anomaly segmentation mask" },
]

export default function ExplainabilityPage() {
  const [selectedView, setSelectedView] = useState<ViewKey>("heatmap")
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

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
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg"] },
    maxFiles: 1,
  })

  const getImageSrc = () => {
    if (!result) return null
    if (selectedView === "heatmap" && result.heatmap_b64) return `data:image/png;base64,${result.heatmap_b64}`
    if (selectedView === "attention" && (result.attention_b64 || result.gradcam_b64)) return `data:image/png;base64,${result.attention_b64 || result.gradcam_b64}`
    if (selectedView === "mask" && result.anomaly_mask_b64) return `data:image/png;base64,${result.anomaly_mask_b64}`
    return null
  }

  const getRationale = () => {
    if (selectedView === "heatmap") return "Regions with high anomaly scores (red) indicate deviations from healthy rail memory bank."
    if (selectedView === "attention") return "Grad-CAM highlights pixels that most influenced the classifier's decision."
    return "Binary mask showing regions classified as anomalous above threshold."
  }

  const imageSrc = getImageSrc()

  return (
    <PageContainer eyebrow="Detect" title="Explainability Viewer" subtitle="Visualizing why defects were detected">
      <div className="mb-6">
        <DemoGallery onPick={(file, preview) => analyze(file, preview)} title="Demo images — click to visualize" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <Card>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">Visualization</h2>
            <div className="space-y-2">
              {VIEWS.map((v) => {
                const Icon = v.icon
                return (
                  <button
                    key={v.key}
                    onClick={() => setSelectedView(v.key)}
                    className={`w-full text-left p-3 rounded-lg transition-colors ${
                      selectedView === v.key ? "bg-rail-500/15 border border-rail-500/40" : "bg-slate-800/50 hover:bg-slate-700"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <Icon size={16} className={selectedView === v.key ? "text-rail-300" : "text-slate-400"} />
                      <p className="font-medium text-sm">{v.label}</p>
                    </div>
                    <p className="text-xs text-slate-400 mt-1 ml-7">{v.desc}</p>
                  </button>
                )
              })}
            </div>
          </Card>

          <Card>
            <h2 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-3">Upload Image</h2>
            <div
              {...getRootProps()}
              className={`aspect-square bg-slate-800/50 rounded-lg border-2 border-dashed flex flex-col items-center justify-center cursor-pointer transition-all ${
                isDragActive ? "border-rail-400 bg-rail-500/15" : "border-slate-600 hover:border-rail-400"
              }`}
            >
              <input {...getInputProps()} />
              {preview ? (
                <img src={preview} alt="Upload" className="w-full h-full object-cover rounded-lg" />
              ) : (
                <>
                  <Upload size={24} className="text-slate-400 mb-2" />
                  <p className="text-xs text-slate-400">Click or drop image</p>
                </>
              )}
            </div>
          </Card>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <Card>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Eye size={18} className="text-rail-300" />
                <h2 className="font-semibold">
                  {VIEWS.find((v) => v.key === selectedView)?.label}
                </h2>
              </div>
              {result && (
                <button className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-800 border border-slate-700 hover:bg-slate-800/50 rounded-lg text-xs transition-colors">
                  <Download size={14} />
                  Export
                </button>
              )}
            </div>

            <div className="aspect-video bg-slate-800/50 rounded-lg border border-slate-700 flex items-center justify-center overflow-hidden">
              {loading ? (
                <div className="text-center">
                  <Loader2 size={32} className="text-rail-300 animate-spin mx-auto mb-2" />
                  <p className="text-sm text-slate-400">Generating visualization...</p>
                </div>
              ) : imageSrc ? (
                <img src={imageSrc} alt={selectedView} className="w-full h-full object-contain" />
              ) : (
                <div className="text-center">
                  <ImageIcon size={40} className="text-slate-600 mx-auto mb-3" />
                  <p className="text-sm text-slate-400">Upload an image to generate visualization</p>
                </div>
              )}
            </div>
          </Card>

          {result && (
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <p className="text-xs text-slate-400 mb-1">Detection Result</p>
                <div className="flex items-center gap-2">
                  <Badge
                    variant={
                      result.status === "defect_detected" ? "error" : result.status === "healthy" ? "success" : "warning"
                    }
                  >
                    {result.status?.replace("_", " ")}
                  </Badge>
                  {result.confidence !== undefined && (
                    <span className="text-sm font-mono text-slate-400">
                      {(result.confidence * 100).toFixed(0)}% confidence
                    </span>
                  )}
                </div>
              </Card>
              <Card>
                <p className="text-xs text-slate-400 mb-1">Research Value</p>
                <p className="text-sm text-slate-200">{getRationale()}</p>
              </Card>
            </div>
          )}
        </div>
      </div>
    </PageContainer>
  )
}
