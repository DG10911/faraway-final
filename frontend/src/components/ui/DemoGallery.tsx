"use client"

import { useEffect, useState } from "react"
import { Images } from "lucide-react"

interface DemoItem { src: string; label: string }

const LABEL_TONE: Record<string, string> = {
  healthy: "text-emerald-400", crack: "text-red-400", squat: "text-orange-400",
  spalling: "text-amber-400", shelling: "text-amber-400", groove: "text-rail-300",
  joint: "text-slate-400", flaking: "text-orange-400",
}

export function DemoGallery({ onPick, title = "Demo images — click to analyze" }: { onPick: (file: File, preview: string) => void; title?: string }) {
  const [items, setItems] = useState<DemoItem[]>([])
  const [busy, setBusy] = useState<string | null>(null)

  useEffect(() => {
    const fallback: DemoItem[] = [
      ...[1, 2, 3].map((n) => ({ src: `/demo/healthy_${n}.jpg`, label: "healthy" })),
      ...[1, 2, 3].map((n) => ({ src: `/demo/crack_${n}.jpg`, label: "crack" })),
      ...[1, 2].map((n) => ({ src: `/demo/squat_${n}.jpg`, label: "squat" })),
      ...[1, 2].map((n) => ({ src: `/demo/spalling_${n}.jpg`, label: "spalling" })),
      ...[1, 2].map((n) => ({ src: `/demo/shelling_${n}.jpg`, label: "shelling" })),
      ...[1, 2].map((n) => ({ src: `/demo/groove_${n}.jpg`, label: "groove" })),
      ...[1, 2].map((n) => ({ src: `/demo/joint_${n}.jpg`, label: "joint" })),
      ...[1, 2].map((n) => ({ src: `/demo/flaking_${n}.jpg`, label: "flaking" })),
    ]
    fetch("/demo/manifest.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((d) => setItems(Array.isArray(d) && d.length ? d : fallback))
      .catch(() => setItems(fallback))
  }, [])

  const pick = async (it: DemoItem) => {
    setBusy(it.src)
    try {
      const res = await fetch(it.src)
      const blob = await res.blob()
      const file = new File([blob], it.src.split("/").pop() || "demo.jpg", { type: blob.type || "image/jpeg" })
      onPick(file, it.src)
    } finally {
      setBusy(null)
    }
  }

  if (!items.length) return null
  return (
    <div className="card-surface p-4">
      <p className="eyebrow mb-2.5 flex items-center gap-1.5"><Images size={13} /> {title}</p>
      <div className="flex gap-3 overflow-x-auto pb-1.5">
        {items.map((it, i) => (
          <button key={i} onClick={() => pick(it)} disabled={!!busy}
            className="group shrink-0 w-[88px] text-left focus:outline-none">
            <div className={`relative w-[88px] h-[58px] rounded-lg overflow-hidden border transition-all ${busy === it.src ? "border-rail-400 opacity-60" : "border-slate-700 group-hover:border-rail-400 group-hover:-translate-y-0.5"}`}>
              <img src={it.src} alt={it.label} className="w-full h-full object-cover" />
            </div>
            <span className={`block mt-1 text-[11px] font-medium capitalize ${LABEL_TONE[it.label] || "text-slate-400"}`}>{it.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
