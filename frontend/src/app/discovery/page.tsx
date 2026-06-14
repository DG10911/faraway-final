"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Search, AlertTriangle, Tag, RefreshCw, FileQuestion, Loader2 } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton"
import { api } from "@/lib/api"
import type { UnknownSample } from "@/lib/types"

export default function DiscoveryPage() {
  const [samples, setSamples] = useState<UnknownSample[]>([])
  const [status, setStatus] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [labeling, setLabeling] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const fetchSamples = useCallback(async () => {
    try {
      const data = await api.unknownSamples()
      setSamples(data)
      setStatus(null)
    } catch {
      setStatus("API offline...")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller
    fetchSamples()
    const interval = setInterval(fetchSamples, 5000)
    return () => {
      controller.abort()
      clearInterval(interval)
    }
  }, [fetchSamples])

  const handleLabel = async (id: string, label: string) => {
    setLabeling(id)
    try {
      await api.labelUnknown(id, label)
      await fetchSamples()
    } catch {
      setStatus("Failed to label sample")
    } finally {
      setLabeling(null)
    }
  }

  return (
    <PageContainer
      title="Unknown Defect Discovery"
      subtitle="Open-set detections that don't match known defect prototypes"
      actions={
        <button
          onClick={fetchSamples}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-800 border border-slate-700 hover:bg-slate-800/50 rounded-lg text-sm transition-colors"
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      }
    >
      <Card className="mb-6">
        <div className="flex items-start gap-3">
          <FileQuestion size={20} className="text-amber-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm font-medium text-amber-400">Research Insight</p>
            <p className="text-sm text-slate-400 mt-1">
              When <code className="text-rail-300">distance &gt; threshold</code> AND{" "}
              <code className="text-rail-300">anomaly_score &gt; threshold</code>, the system returns{" "}
              <code className="text-rail-300">unknown_anomaly</code> instead of forcing assignment to known classes.
            </p>
          </div>
        </div>
      </Card>

      {status && (
        <div className="flex items-center gap-3 p-4 mb-6 rounded-lg bg-amber-500/15 border border-amber-500/30 text-amber-400 text-sm">
          <AlertTriangle size={16} className="shrink-0" />
          <span>{status}</span>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <LoadingSkeleton key={i} variant="card" />
          ))}
        </div>
      ) : samples.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {samples.map((s) => (
            <Card key={s.id} className="border border-amber-500/30">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <AlertTriangle size={16} className="text-amber-400" />
                  <span className="font-mono text-amber-400 text-sm">{s.id}</span>
                </div>
                <Badge variant="warning">Unknown</Badge>
              </div>

              <div className="grid grid-cols-3 gap-3 text-sm mb-4">
                <div>
                  <span className="text-slate-400 text-xs">Confidence</span>
                  <p className="font-mono">{(s.anomaly_score * 100).toFixed(0)}%</p>
                </div>
                <div>
                  <span className="text-slate-400 text-xs">Track</span>
                  <p className="font-mono">{s.track_id}</p>
                </div>
                <div>
                  <span className="text-slate-400 text-xs">Distance</span>
                  <p className="font-mono">{s.distance?.toFixed(3) ?? "—"}</p>
                </div>
              </div>

              <div className="flex gap-2">
                {["crack", "squat", "spalling"].map((label) => (
                  <button
                    key={label}
                    onClick={() => handleLabel(s.id, label)}
                    disabled={labeling === s.id}
                    className="flex items-center gap-1 px-2.5 py-1 text-xs bg-rail-500/15 text-rail-300 rounded-lg hover:bg-rail-500/20 transition-colors disabled:opacity-50"
                  >
                    {labeling === s.id ? <Loader2 size={12} className="animate-spin" /> : <Tag size={12} />}
                    {label}
                  </button>
                ))}
              </div>
            </Card>
          ))}
        </div>
      ) : (
        <Card>
          <div className="flex flex-col items-center justify-center py-12 text-slate-400">
            <Search size={32} className="mb-3" />
            <p>No unknown anomalies flagged yet</p>
            <p className="text-xs mt-1">New defect categories will appear here automatically</p>
          </div>
        </Card>
      )}
    </PageContainer>
  )
}
