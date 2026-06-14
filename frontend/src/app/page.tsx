"use client"

import { useState, useEffect, useRef } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import {
  Radio, Upload, Search, Beaker, Wrench, Eye, AlertCircle, Activity,
  ArrowRight, ShieldCheck, Layers, Sparkles, Cpu,
} from "lucide-react"
import Link from "next/link"
import { Card } from "@/components/ui/Card"
import { StatusDot } from "@/components/ui/StatusDot"
import { api } from "@/lib/api"

const NAV_CARDS = [
  { label: "Live Monitoring", href: "/live", icon: Radio, desc: "Real-time rail inspection feed with rolling alerts." },
  { label: "Upload Detection", href: "/upload", icon: Upload, desc: "Analyze a single rail image end-to-end." },
  { label: "Unknown Discovery", href: "/discovery", icon: Search, desc: "Label novel defects → instant few-shot class." },
  { label: "Few-Shot Lab", href: "/few-shot", icon: Beaker, desc: "Episodic evaluation with mean ± 95% CI." },
  { label: "Safety & Calibration", href: "/safety", icon: ShieldCheck, desc: "Conformal recall, cross-domain, augmentation." },
  { label: "Maintenance", href: "/maintenance", icon: Wrench, desc: "Digital-twin priorities by failure risk." },
  { label: "Explainability", href: "/explain", icon: Eye, desc: "Anomaly heatmaps & DINOv2 attention." },
]

const PIPELINE_STEPS = [
  { t: "Self-supervised features", d: "Frozen DINOv2 dense patch tokens", icon: Cpu },
  { t: "Healthy-only screening", d: "PatchCore memory bank · zero defect labels", icon: ShieldCheck },
  { t: "Few-shot naming", d: "Prototypical network · 5 examples per class", icon: Sparkles },
  { t: "Open-set discovery", d: "Reject unknown → label → adapt live", icon: Search },
  { t: "Conformal recall guarantee", d: "Distribution-free safety bound", icon: ShieldCheck },
  { t: "Digital-twin risk ranking", d: "Prioritized predictive maintenance", icon: Layers },
]

interface Kpi { label: string; value: number | string; tone: string }

export default function Home() {
  const [status, setStatus] = useState<string>("connecting…")
  const [initialized, setInitialized] = useState(false)
  const [distribution, setDistribution] = useState<{ name: string; count: number }[]>([])
  const [kpis, setKpis] = useState<Kpi[]>([])
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
        if (!controller.signal.aborted) { setStatus("API Offline"); setInitialized(false) }
      }
      try {
        const s = await api.stats()
        if (!controller.signal.aborted)
          setKpis([
            { label: "Frames processed", value: s.frames_processed ?? 0, tone: "text-rail-300" },
            { label: "Anomalies found", value: s.anomalies_found ?? 0, tone: "text-cyan-400" },
            { label: "Defects confirmed", value: s.defects_confirmed ?? 0, tone: "text-danger-400" },
            { label: "Unknowns flagged", value: s.unknowns_flagged ?? 0, tone: "text-amber-400" },
          ])
      } catch { /* ignore */ }
      try {
        const dd = await api.defectDistribution()
        if (!controller.signal.aborted)
          setDistribution(
            Object.entries(dd.distribution || {}).map(([name, count]) => ({
              name: name.charAt(0).toUpperCase() + name.slice(1), count: count as number,
            }))
          )
      } catch { /* ignore */ }
    }
    load()
    const interval = setInterval(load, 5000)
    return () => { controller.abort(); clearInterval(interval) }
  }, [])

  const barColors = ["#4a45e6", "#5b63f5", "#0d9488", "#06b6d4", "#7c87ff", "#3a34c2"]

  return (
    <div className="relativez min-h-screen p-4 lg:p-8 max-w-[1400px] mx-auto">
      {/* HERO */}
      <section className="relative overflow-hidden card-surface p-8 lg:p-12 mb-6 animate-fade-up">
        <div className="absolute top-0 right-0 h-full w-1/3 stat-grad pointer-events-none hidden lg:block" style={{ clipPath: "polygon(30% 0, 100% 0, 100% 100%, 0% 100%)" }} />
        <div className="relative max-w-3xl">
          <div className="flex flex-wrap items-center gap-3 mb-5">
            <span className="chip"><Sparkles size={13} /> Few-Shot Rail Defect Intelligence</span>
            <StatusDot status={initialized ? "online" : status === "API Offline" ? "offline" : "warning"} label={status} />
          </div>
          <h1 className="text-4xl lg:text-5xl font-extrabold tracking-tight leading-[1.06] text-slate-100">
            Detect with <span className="gradient-text">zero</span> examples.
            <br />Name with <span className="gradient-text">five</span>. Guarantee the recall.
          </h1>
          <p className="text-slate-400 mt-5 text-base lg:text-lg leading-relaxed">
            A research-grade platform fusing DINOv2 self-supervised features, healthy-only PatchCore screening,
            prototypical few-shot classification, open-set discovery, and conformal safety — built for the reality
            that labelled rail defects are vanishingly rare.
          </p>
          <div className="flex flex-wrap items-center gap-3 mt-7">
            <Link href="/upload" className="btn btn-primary">Run a detection <ArrowRight size={16} /></Link>
            <Link href="/few-shot" className="btn btn-ghost">Open Few-Shot Lab</Link>
            <Link href="/discovery" className="btn btn-ghost">Discovery flywheel</Link>
          </div>
        </div>
      </section>

      {/* KPI STRIP */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        {(kpis.length ? kpis : [
          { label: "Frames processed", value: "—", tone: "text-rail-300" },
          { label: "Anomalies found", value: "—", tone: "text-cyan-400" },
          { label: "Defects confirmed", value: "—", tone: "text-danger-400" },
          { label: "Unknowns flagged", value: "—", tone: "text-amber-400" },
        ]).map((k) => (
          <div key={k.label} className="card-surface p-5">
            <p className="text-xs text-slate-400">{k.label}</p>
            <p className={`mt-1 text-3xl font-bold tabular-nums ${k.tone}`}>{k.value}</p>
          </div>
        ))}
      </div>

      {/* FEATURE GRID */}
      <h2 className="eyebrow mb-3">Explore the platform</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-10">
        {NAV_CARDS.map((item) => {
          const Icon = item.icon
          return (
            <Link key={item.href} href={item.href}>
              <Card hover className="h-full group">
                <div className="flex items-start justify-between">
                  <div className="w-11 h-11 rounded-xl bg-brand-soft border border-rail-500/20 flex items-center justify-center mb-4 group-hover:scale-105 transition-transform">
                    <Icon size={20} className="text-rail-300" />
                  </div>
                  <ArrowRight size={18} className="text-slate-600 group-hover:text-rail-300 group-hover:translate-x-1 transition-all" />
                </div>
                <h3 className="font-semibold text-slate-100 group-hover:text-rail-300 transition-colors">{item.label}</h3>
                <p className="text-sm text-slate-400 mt-1 leading-relaxed">{item.desc}</p>
              </Card>
            </Link>
          )
        })}
      </div>

      {/* CHART + PIPELINE */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <Card className="lg:col-span-3">
          <div className="flex items-center gap-2 mb-5">
            <Activity size={18} className="text-rail-300" />
            <h2 className="text-lg font-semibold text-slate-100">Defect Distribution</h2>
          </div>
          {distribution.length > 0 ? (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={distribution}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2840" vertical={false} />
                <XAxis dataKey="name" stroke="#7888b8" fontSize={12} tickLine={false} axisLine={{ stroke: "#2a3550" }} />
                <YAxis stroke="#7888b8" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                <Tooltip
                  cursor={{ fill: "rgba(124,135,255,0.12)" }}
                  contentStyle={{ backgroundColor: "#121829", border: "1px solid #232c41", borderRadius: 12, boxShadow: "0 16px 40px -16px rgba(0,0,0,0.7)" }}
                  labelStyle={{ color: "#e7eaf3" }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {distribution.map((_, i) => <Cell key={i} fill={barColors[i % barColors.length]} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div className="flex flex-col items-center justify-center h-[280px] text-slate-500">
              <AlertCircle size={26} className="mb-2 text-slate-600" />
              <p className="text-sm">No detections yet — run one from the Upload page.</p>
            </div>
          )}
        </Card>

        <Card className="lg:col-span-2">
          <div className="flex items-center gap-2 mb-5">
            <Layers size={18} className="text-rail-300" />
            <h2 className="text-lg font-semibold text-slate-100">The Pipeline</h2>
          </div>
          <ol className="relative space-y-4">
            <span className="absolute left-[19px] top-2 bottom-2 w-px bg-gradient-to-b from-rail-400 to-accent-400" />
            {PIPELINE_STEPS.map((s, i) => {
              const Icon = s.icon
              return (
                <li key={i} className="relative flex items-start gap-3">
                  <span className="z-[1] w-10 h-10 shrink-0 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center">
                    <Icon size={16} className="text-rail-300" />
                  </span>
                  <div className="pt-0.5">
                    <p className="text-sm font-semibold text-slate-100">{s.t}</p>
                    <p className="text-xs text-slate-400">{s.d}</p>
                  </div>
                </li>
              )
            })}
          </ol>
        </Card>
      </div>
    </div>
  )
}
