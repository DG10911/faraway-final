"use client"

import { useState, useEffect, useRef } from "react"
import { Activity, AlertTriangle, Clock, Cpu, Zap, AlertCircle, Info } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { StatusDot } from "@/components/ui/StatusDot"
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton"
import { api } from "@/lib/api"
import type { Stats } from "@/lib/types"

export default function LiveMonitoring() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [offline, setOffline] = useState(false)
  const [loading, setLoading] = useState(true)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller

    const load = async () => {
      try {
        const s = await api.stats()
        if (!controller.signal.aborted) {
          setStats(s)
          setOffline(false)
        }
      } catch {
        if (!controller.signal.aborted) setOffline(true)
      } finally {
        if (!controller.signal.aborted) setLoading(false)
      }
    }

    load()
    const interval = setInterval(load, 2000)
    return () => {
      controller.abort()
      clearInterval(interval)
    }
  }, [])

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
      subtitle="Real-time rail inspection pipeline metrics"
      actions={
        <StatusDot
          status={offline ? "offline" : stats?.initialized ? "online" : "warning"}
          label={offline ? "Disconnected" : stats?.initialized ? "Model Ready" : "Standby"}
        />
      }
    >
      {offline && (
        <div className="flex items-center gap-3 p-4 mb-6 rounded-lg bg-red-500/15 border border-red-500/30 text-red-400 text-sm">
          <AlertCircle size={18} className="shrink-0" />
          <span>API is offline. Reconnecting...</span>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <LoadingSkeleton key={i} variant="card" />
          ))}
        </div>
      ) : (
        <>
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
              {stats?.throughput_fps !== null ? (
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl font-bold font-mono text-emerald-400">{stats?.throughput_fps?.toFixed(1)}</span>
                  <span className="text-slate-400">fps</span>
                </div>
              ) : (
                <p className="text-sm text-slate-400">Waiting for data...</p>
              )}
              <div className="mt-4 h-2 bg-slate-700 rounded-full overflow-hidden">
                <div
                  className="h-full bg-green-500 rounded-full transition-all duration-1000"
                  style={{ width: `${Math.min((stats?.throughput_fps ?? 0) / 60 * 100, 100)}%` }}
                />
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
        </>
      )}
    </PageContainer>
  )
}
