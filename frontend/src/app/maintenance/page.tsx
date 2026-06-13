"use client"

import { useState, useEffect, useRef } from "react"
import { AlertTriangle, Activity, Shield, TrendingUp, MapPin, Hash, Gauge } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton"
import { api } from "@/lib/api"
import type { Segment, Ranking } from "@/lib/types"

const TRACK_IDS = ["Track_A12", "Track_B7", "Track_C3", "Track_D9"]

const SEVERITY_BADGE: Record<string, "error" | "warning" | "success" | "info"> = {
  Critical: "error",
  High: "warning",
  Medium: "warning",
  Low: "success",
}

export default function MaintenancePage() {
  const [segments, setSegments] = useState<Segment[]>([])
  const [offline, setOffline] = useState(false)
  const [loading, setLoading] = useState(true)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller

    const load = async () => {
      try {
        const results = await Promise.all(TRACK_IDS.map((id) => api.twinStatus(id)))
        if (!controller.signal.aborted) {
          setSegments(results.filter(Boolean))
          setOffline(false)
        }
      } catch {
        if (!controller.signal.aborted) setOffline(true)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    load()
    const interval = setInterval(load, 5000)
    return () => {
      controller.abort()
      clearInterval(interval)
    }
  }, [])

  const allRankings: Ranking[] = segments.flatMap((s) => s.priority_rankings ?? []).sort(
    (a, b) => b.failure_risk_pct - a.failure_risk_pct
  )

  return (
    <PageContainer
      title="Maintenance Priorities"
      subtitle="Digital twin predictive maintenance intelligence"
    >
      {offline && (
        <div className="flex items-center gap-3 p-4 mb-6 rounded-lg bg-red-900/20 border border-red-800/40 text-red-400 text-sm">
          <AlertTriangle size={18} className="shrink-0" />
          <span>API is offline. Showing cached data.</span>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <LoadingSkeleton variant="card" />
          <LoadingSkeleton variant="card" />
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <Card>
            <div className="flex items-center gap-2 mb-4">
              <MapPin size={18} className="text-rail-400" />
              <h2 className="font-semibold">Railway Segments</h2>
            </div>
            <div className="space-y-3">
              {segments.length > 0 ? (
                segments.map((seg) => (
                  <div key={seg.track_id} className="bg-gray-900 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <span className="font-semibold">{seg.track_id.replace("_", " ")}</span>
                      <Badge variant={seg.overall_health > 0.7 ? "success" : seg.overall_health > 0.4 ? "warning" : "error"}>
                        {seg.overall_health > 0.7 ? "Good" : seg.overall_health > 0.4 ? "At Risk" : "Critical"}
                      </Badge>
                    </div>
                    <div className="mt-3 flex items-center gap-4 text-sm">
                      <span className="text-gray-400">{seg.active_defects} active defects</span>
                      <span className="flex items-center gap-1 text-gray-400">
                        <Gauge size={14} />
                        Health {(seg.overall_health * 100).toFixed(0)}%
                      </span>
                    </div>
                    <div className="mt-2 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all ${
                          seg.overall_health > 0.7 ? "bg-green-500" : seg.overall_health > 0.4 ? "bg-yellow-500" : "bg-red-500"
                        }`}
                        style={{ width: `${seg.overall_health * 100}%` }}
                      />
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500 py-4 text-center">No segments found</p>
              )}
            </div>
          </Card>

          <Card>
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp size={18} className="text-rail-400" />
              <h2 className="font-semibold">Priority Rankings</h2>
            </div>
            <div className="space-y-3">
              {allRankings.length > 0 ? (
                allRankings.map((r) => (
                  <div key={r.event_id} className="flex items-center justify-between bg-gray-900 rounded-lg p-3">
                    <div className="flex items-center gap-3">
                      <span className="w-8 h-8 rounded-full bg-rail-900 text-rail-300 flex items-center justify-center text-sm font-bold">
                        {r.priority_rank}
                      </span>
                      <div>
                        <p className="font-medium text-sm">{r.track_id.replace("_", " ")}</p>
                        <p className="text-xs text-gray-400">{r.defect_type.replace("_", " ")}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <Badge variant={SEVERITY_BADGE[r.severity] || "default"}>{r.severity}</Badge>
                      <p className="text-sm font-mono text-danger-400 mt-1">{r.failure_risk_pct}% risk</p>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-gray-500 py-4 text-center">No defects reported yet</p>
              )}
            </div>
          </Card>
        </div>
      )}

      <Card>
        <div className="flex items-center gap-2 mb-4">
          <Shield size={18} className="text-rail-400" />
          <h2 className="font-semibold">Digital Twin Logic</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-900 rounded-lg p-3">
            <p className="flex items-center gap-1.5 text-rail-400 font-medium mb-1">
              <Hash size={14} /> Failure Risk
            </p>
            <p className="text-gray-400 text-xs">anomaly_score × 0.6 + severity_weight × 0.4</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <p className="flex items-center gap-1.5 text-rail-400 font-medium mb-1">
              <AlertTriangle size={14} /> Severity Weights
            </p>
            <p className="text-gray-400 text-xs">Low=0.2 Med=0.4 High=0.7 Crit=0.95</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <p className="flex items-center gap-1.5 text-rail-400 font-medium mb-1">
              <Activity size={14} /> Priority
            </p>
            <p className="text-gray-400 text-xs">Sorted by failure risk descending</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <p className="flex items-center gap-1.5 text-rail-400 font-medium mb-1">
              <Gauge size={14} /> Health Score
            </p>
            <p className="text-gray-400 text-xs">1.0 — avg failure risk across defects</p>
          </div>
        </div>
      </Card>
    </PageContainer>
  )
}
