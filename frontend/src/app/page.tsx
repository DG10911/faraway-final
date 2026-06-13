"use client"

import { useState, useEffect, useRef } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import { LayoutDashboard, Radio, Upload, Search, Beaker, Wrench, Eye, AlertCircle, Activity } from "lucide-react"
import Link from "next/link"
import { Card } from "@/components/ui/Card"
import { StatusDot } from "@/components/ui/StatusDot"
import { LoadingSkeleton } from "@/components/ui/LoadingSkeleton"
import { api } from "@/lib/api"

const NAV_CARDS = [
  { label: "Live Monitoring", href: "/live", icon: Radio, desc: "Real-time rail inspection feed" },
  { label: "Upload Detection", href: "/upload", icon: Upload, desc: "Analyze a single rail image" },
  { label: "Unknown Discovery", href: "/discovery", icon: Search, desc: "Novel defect categories" },
  { label: "Few-Shot Lab", href: "/few-shot", icon: Beaker, desc: "Episodic training & evaluation" },
  { label: "Maintenance", href: "/maintenance", icon: Wrench, desc: "Digital twin priorities" },
  { label: "Explainability", href: "/explain", icon: Eye, desc: "Grad-CAM & attention maps" },
]

const PIPELINE_STEPS = [
  "Self-Supervised Representation Learning (DINOv2)",
  "Healthy-Only Anomaly Detection (PatchCore)",
  "Few-Shot Classification (Prototypical Networks)",
  "Open-Set Unknown Defect Discovery",
  "Cross-Domain Adaptation (Mendeley → RSDDs)",
  "Predictive Maintenance Digital Twin",
]

export default function Home() {
  const [status, setStatus] = useState<string>("connecting...")
  const [initialized, setInitialized] = useState(false)
  const [distribution, setDistribution] = useState<{ name: string; count: number }[]>([])
  const [loading, setLoading] = useState(true)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    abortRef.current = controller

    const load = async () => {
      try {
        const health = await api.health()
        if (!controller.signal.aborted) {
          setStatus(health.initialized ? "Model Ready" : "Not Initialized")
          setInitialized(health.initialized)
        }
      } catch {
        if (!controller.signal.aborted) {
          setStatus("API Offline")
          setInitialized(false)
        }
      }
      try {
        const dd = await api.defectDistribution()
        if (!controller.signal.aborted) {
          setDistribution(
            Object.entries(dd.distribution || {}).map(([name, count]) => ({
              name: name.charAt(0).toUpperCase() + name.slice(1),
              count: count as number,
            }))
          )
        }
      } catch {
        /* ignore */
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

  return (
    <div className="min-h-screen p-4 lg:p-6">
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              RailGuard<span className="text-rail-400">-FSL++</span>
            </h1>
            <p className="text-gray-400 mt-1 text-sm">Few-Shot Defect Detection in Rail Infrastructure</p>
          </div>
          <div className="flex items-center gap-3">
            <StatusDot
              status={initialized ? "online" : status === "API Offline" ? "offline" : "warning"}
              label={status}
            />
          </div>
        </div>
      </header>

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {Array.from({ length: 6 }).map((_, i) => (
            <LoadingSkeleton key={i} variant="card" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
          {NAV_CARDS.map((item) => {
            const Icon = item.icon
            return (
              <Link key={item.href} href={item.href}>
                <Card className="h-full hover:border-rail-500/50 transition-all group cursor-pointer">
                  <div className="w-10 h-10 rounded-lg bg-rail-900/60 flex items-center justify-center mb-3 group-hover:bg-rail-800/60 transition-colors">
                    <Icon size={20} className="text-rail-400" />
                  </div>
                  <h3 className="font-semibold group-hover:text-rail-400 transition-colors">{item.label}</h3>
                  <p className="text-sm text-gray-500 mt-1">{item.desc}</p>
                </Card>
              </Link>
            )
          })}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Activity size={18} className="text-rail-400" />
            <h2 className="text-lg font-semibold">Defect Distribution</h2>
          </div>
          {distribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Bar dataKey="count" fill="#0c8ee7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex flex-col items-center justify-center h-48 text-gray-500">
              <AlertCircle size={24} className="mb-2" />
              <p className="text-sm">No detections yet</p>
            </div>
          )}
        </Card>

        <Card>
          <div className="flex items-center gap-2 mb-4">
            <LayoutDashboard size={18} className="text-rail-400" />
            <h2 className="text-lg font-semibold">Research Pipeline</h2>
          </div>
          <div className="space-y-3">
            {PIPELINE_STEPS.map((step, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="w-6 h-6 rounded-full bg-rail-900 text-rail-300 flex items-center justify-center text-xs font-bold shrink-0">
                  {i + 1}
                </span>
                <span className="text-gray-300">{step}</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  )
}
