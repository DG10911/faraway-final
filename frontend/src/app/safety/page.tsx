"use client"

import { useState, useEffect, useCallback } from "react"
import { useDropzone } from "react-dropzone"
import { ShieldCheck, SlidersHorizontal, Sparkles, Play, Loader2, Globe, Upload, Image as ImageIcon } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { DemoGallery } from "@/components/ui/DemoGallery"
import { api } from "@/lib/api"
import type { ConformalResult, DomainCalibration, AugmentResult } from "@/lib/types"

function Stat({ label, value, tone = "text-slate-100" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-800/50 px-3.5 py-3">
      <p className="text-xs text-slate-400">{label}</p>
      <p className={`text-xl font-bold font-mono mt-0.5 ${tone}`}>{value}</p>
    </div>
  )
}

const pct = (v?: number | null) => (v === undefined || v === null ? "—" : `${(v * 100).toFixed(1)}%`)

export default function SafetyPage() {
  // ---- conformal ----
  const [recall, setRecall] = useState(0.95)
  const [conf, setConf] = useState<ConformalResult | null>(null)
  const [confScores, setConfScores] = useState<{ n_defect: number; n_healthy: number }>({ n_defect: 0, n_healthy: 0 })
  const [confLoading, setConfLoading] = useState(false)
  const [confErr, setConfErr] = useState<string | null>(null)

  // ---- cross-domain ----
  const [percentile, setPercentile] = useState(95)
  const [domain, setDomain] = useState("Mendeley")
  const [domains, setDomains] = useState<DomainCalibration[]>([])
  const [domLoading, setDomLoading] = useState(false)

  // ---- augmentation ----
  const [kind, setKind] = useState("crack")
  const [aug, setAug] = useState<AugmentResult | null>(null)
  const [augLoading, setAugLoading] = useState(false)

  useEffect(() => {
    api.conformalStatus().then((s) => { setConf(s.conformal); setConfScores(s.scores) }).catch(() => {})
    api.calibrateDomains().then((d) => setDomains(d.domains || [])).catch(() => {})
  }, [])

  const runConformal = async () => {
    setConfLoading(true); setConfErr(null)
    try { setConf(await api.conformalCalibrate(recall)) }
    catch (e: any) { setConfErr(String(e?.message || e).replace(/^API \d+:\s*/, "")) }
    finally { setConfLoading(false) }
  }

  const runDomain = async () => {
    setDomLoading(true)
    try {
      await api.calibrateThreshold(percentile, domain)
      const d = await api.calibrateDomains(); setDomains(d.domains || [])
    } catch { /* offline */ } finally { setDomLoading(false) }
  }

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0]; if (!file) return
    setAugLoading(true); setAug(null)
    try { setAug(await api.augmentPreview(file, kind)) }
    catch { setAug(null) } finally { setAugLoading(false) }
  }, [kind])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "image/*": [".png", ".jpg", ".jpeg"] }, maxFiles: 1 })

  return (
    <PageContainer
      eyebrow="Assurance"
      title="Safety & Calibration"
      subtitle="The research-grade guarantees: a distribution-free recall bound, per-domain threshold calibration, and synthetic defect augmentation for the scarce-positive regime."
    >
      {/* CONFORMAL */}
      <Card className="mb-6">
        <div className="flex items-center gap-2 mb-1.5">
          <ShieldCheck size={18} className="text-rail-300" />
          <h2 className="text-lg font-semibold text-slate-100">Conformal recall guarantee</h2>
          <Badge variant="info" className="ml-1">distribution-free</Badge>
        </div>
        <p className="text-sm text-slate-400 mb-5 max-w-3xl">
          Pick the anomaly threshold from a held-out set of defect scores so that
          <code className="mx-1 text-rail-300">P(defect detected) ≥ target</code>
          holds with a finite-sample guarantee (split conformal prediction). Collected scores so far:
          <span className="font-mono text-slate-200"> {confScores.n_defect} defect · {confScores.n_healthy} healthy</span>.
        </p>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div>
            <label className="flex items-center justify-between text-sm mb-1.5">
              <span className="text-slate-400">Target recall</span>
              <Badge variant="info">{(recall * 100).toFixed(0)}%</Badge>
            </label>
            <input type="range" min={0.8} max={0.99} step={0.01} value={recall}
              onChange={(e) => setRecall(Number(e.target.value))} className="w-full accent-rail-400" />
            <div className="flex justify-between text-xs text-slate-500 mt-1"><span>80%</span><span>90%</span><span>99%</span></div>
            <button onClick={runConformal} disabled={confLoading}
              className="btn btn-primary w-full mt-4">
              {confLoading ? <><Loader2 size={16} className="animate-spin" /> Calibrating…</> : <><Play size={16} /> Calibrate guarantee</>}
            </button>
            {confErr && <p className="text-xs text-amber-400 mt-2">{confErr}</p>}
          </div>
          <div className="lg:col-span-2">
            {conf && !conf.error ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Guaranteed recall" value={pct(conf.guaranteed_recall)} tone="text-rail-300" />
                <Stat label="Empirical recall" value={pct(conf.empirical_recall)} tone="text-emerald-400" />
                <Stat label="False-positive rate" value={pct(conf.false_positive_rate)} tone="text-danger-400" />
                <Stat label="Calibration defects" value={`${conf.n_calibration}`} />
                <div className="col-span-2 md:col-span-4 text-xs text-slate-400 rounded-xl border border-slate-700 bg-slate-800 px-3.5 py-2.5">
                  {conf.guarantee_achievable
                    ? <>Guarantee active{conf.applied ? " and applied to the live detector" : ""}: {conf.note}</>
                    : <span className="text-amber-400">{conf.note}</span>}
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full min-h-[140px] text-slate-500 rounded-xl border border-dashed border-slate-700">
                <ShieldCheck size={26} className="mb-2 text-slate-600" />
                <p className="text-sm">Run a few detections, then calibrate to see the recall bound.</p>
              </div>
            )}
          </div>
        </div>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CROSS-DOMAIN */}
        <Card>
          <div className="flex items-center gap-2 mb-1.5">
            <SlidersHorizontal size={18} className="text-rail-300" />
            <h2 className="text-lg font-semibold text-slate-100">Cross-domain calibration</h2>
          </div>
          <p className="text-sm text-slate-400 mb-5">
            Each capture domain (railway, camera, lighting) shifts the score scale. Re-pick the operating threshold
            at a healthy percentile per domain — no retraining.
          </p>
          <div className="space-y-4">
            <div className="flex gap-3">
              <input value={domain} onChange={(e) => setDomain(e.target.value)} placeholder="Domain name"
                className="flex-1 rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-rail-500/40" />
              <Badge variant="default" className="self-center">p{percentile}</Badge>
            </div>
            <input type="range" min={90} max={99.5} step={0.5} value={percentile}
              onChange={(e) => setPercentile(Number(e.target.value))} className="w-full accent-rail-400" />
            <div className="flex justify-between text-xs text-slate-500"><span>90</span><span>95</span><span>99.5</span></div>
            <button onClick={runDomain} disabled={domLoading} className="btn btn-ghost w-full">
              {domLoading ? <><Loader2 size={16} className="animate-spin" /> Recalibrating…</> : <><Globe size={16} /> Recalibrate threshold</>}
            </button>
          </div>
          {domains.length > 0 && (
            <div className="mt-5 space-y-2">
              {domains.map((d) => (
                <div key={d.domain} className="flex items-center justify-between rounded-xl border border-slate-700 bg-slate-800/50 px-3.5 py-2.5 text-sm">
                  <span className="font-medium text-slate-100">{d.domain}</span>
                  <span className="flex items-center gap-3 font-mono text-xs text-slate-400">
                    <span>thr {d.threshold.toFixed(3)}</span>
                    <span className="text-danger-400">fpr {pct(d.false_positive_rate)}</span>
                    <Badge variant="default">p{d.percentile}</Badge>
                  </span>
                </div>
              ))}
            </div>
          )}
        </Card>

        {/* AUGMENTATION */}
        <Card>
          <div className="flex items-center gap-2 mb-1.5">
            <Sparkles size={18} className="text-rail-300" />
            <h2 className="text-lg font-semibold text-slate-100">Synthetic augmentation</h2>
            <Badge variant="default" className="ml-1">CutPaste · procedural</Badge>
          </div>
          <p className="text-sm text-slate-400 mb-4">
            Expand a scarce defect class by synthesizing new examples on healthy crops. GAN/diffusion generators
            (DefectGAN, AnomalyDiffusion) plug into the same interface.
          </p>
          <div className="flex flex-wrap gap-2 mb-3">
            {["crack", "squat", "spalling", "groove"].map((k) => (
              <button key={k} onClick={() => setKind(k)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors capitalize ${
                  kind === k ? "bg-rail-500/15 border-rail-500/30 text-rail-300" : "bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-800/50"
                }`}>{k}</button>
            ))}
          </div>
          <div className="mb-3"><DemoGallery onPick={(f) => onDrop([f])} title="Demo crops — click to synthesize" /></div>
          <div {...getRootProps()}
            className={`rounded-xl border-2 border-dashed p-5 text-center cursor-pointer transition-all ${
              isDragActive ? "border-rail-400 bg-rail-500/15" : "border-slate-600 hover:border-rail-400 bg-slate-800/50"
            }`}>
            <input {...getInputProps()} />
            <Upload size={20} className="mx-auto text-rail-300 mb-1.5" />
            <p className="text-sm text-slate-300">Drop a healthy rail crop to synthesize a <span className="font-semibold capitalize">{kind}</span></p>
          </div>
          <div className="grid grid-cols-2 gap-3 mt-4">
            <div>
              <p className="eyebrow mb-1.5">Original</p>
              <div className="aspect-square rounded-xl border border-slate-700 bg-slate-800/50 flex items-center justify-center overflow-hidden">
                {aug?.original_b64 ? <img src={`data:image/png;base64,${aug.original_b64}`} className="w-full h-full object-cover" alt="original" />
                  : <ImageIcon size={26} className="text-slate-600" />}
              </div>
            </div>
            <div>
              <p className="eyebrow mb-1.5">Synthetic {kind}</p>
              <div className="aspect-square rounded-xl border border-slate-700 bg-slate-800/50 flex items-center justify-center overflow-hidden">
                {augLoading ? <Loader2 size={24} className="text-rail-300 animate-spin" />
                  : aug?.synthetic_b64 ? <img src={`data:image/png;base64,${aug.synthetic_b64}`} className="w-full h-full object-cover" alt="synthetic" />
                  : <Sparkles size={26} className="text-slate-600" />}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </PageContainer>
  )
}
