"use client"

import { useState } from "react"
import { Beaker, Play, RotateCcw, BarChart3, TrendingUp, Layers, Target, Loader2 } from "lucide-react"
import { PageContainer } from "@/components/layout/PageContainer"
import { Card } from "@/components/ui/Card"
import { Badge } from "@/components/ui/Badge"
import { api } from "@/lib/api"
import type { FewShotResult } from "@/lib/types"

export default function FewShotLab() {
  const [nShots, setNShots] = useState(5)
  const [nWays, setNWays] = useState(5)
  const [nEpisodes, setNEpisodes] = useState(100)
  const [result, setResult] = useState<FewShotResult | null>(null)
  const [loading, setLoading] = useState(false)

  const runEval = async () => {
    setLoading(true)
    setResult(null)
    try {
      const data = await api.evaluateFewShot(nWays, nShots, nEpisodes)
      setResult(data)
    } catch {
      setResult({ mean_accuracy: 0, std_accuracy: 0, n_episodes: 0, error: "API unreachable" })
    } finally {
      setLoading(false)
    }
  }

  return (
    <PageContainer
      title="Few-Shot Learning Lab"
      subtitle="Episodic training with median prototypes"
    >
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Layers size={18} className="text-rail-300" />
            <h2 className="font-semibold">Configuration</h2>
          </div>
          <div className="space-y-5">
            <div>
              <label className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-slate-400">N-Shot</span>
                <Badge variant="info">{nShots}-shot</Badge>
              </label>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={nShots}
                onChange={(e) => setNShots(Number(e.target.value))}
                className="w-full accent-rail-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>1</span><span>3</span><span>5</span><span>10</span>
              </div>
            </div>

            <div>
              <label className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-slate-400">N-Way</span>
                <Badge variant="info">{nWays}-way</Badge>
              </label>
              <input
                type="range"
                min={2}
                max={7}
                step={1}
                value={nWays}
                onChange={(e) => setNWays(Number(e.target.value))}
                className="w-full accent-rail-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>2</span><span>3</span><span>5</span><span>7</span>
              </div>
            </div>

            <div>
              <label className="flex items-center justify-between text-sm mb-1.5">
                <span className="text-slate-400">Episodes</span>
                <span className="font-mono text-xs text-slate-400">{nEpisodes}</span>
              </label>
              <input
                type="range"
                min={10}
                max={500}
                step={10}
                value={nEpisodes}
                onChange={(e) => setNEpisodes(Number(e.target.value))}
                className="w-full accent-rail-500"
              />
              <div className="flex justify-between text-xs text-slate-500 mt-1">
                <span>10</span><span>100</span><span>250</span><span>500</span>
              </div>
            </div>

            <button
              onClick={runEval}
              disabled={loading}
              className="w-full flex items-center justify-center gap-2 py-2.5 bg-rail-600 hover:bg-rail-700 text-white rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {loading ? (
                <><Loader2 size={16} className="animate-spin" /> Running...</>
              ) : (
                <><Play size={16} /> Run Evaluation</>
              )}
            </button>
          </div>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          <Card>
            <div className="flex items-center gap-2 mb-4">
              <BarChart3 size={18} className="text-rail-300" />
              <h2 className="font-semibold">Results</h2>
            </div>

            {result ? (
              result.error ? (
                <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                  <RotateCcw size={28} className="mb-3" />
                  <p>Error: {result.error}</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {[
                    { label: "Mean Accuracy", value: `${(result.mean_accuracy * 100).toFixed(1)}%`, icon: Target, color: "text-rail-300" },
                    { label: "Std Dev", value: `${(result.std_accuracy * 100).toFixed(1)}%`, icon: TrendingUp, color: "text-slate-400" },
                    { label: "95% CI", value: result.ci_95 ? `±${(result.ci_95 * 100).toFixed(1)}%` : "—", icon: BarChart3, color: "text-slate-400" },
                    { label: "Open-Set Detection", value: result.open_set_detection_rate !== undefined ? `${(result.open_set_detection_rate * 100).toFixed(1)}%` : "N/A", icon: Layers, color: "text-amber-400" },
                  ].map((m, i) => {
                    const Icon = m.icon
                    return (
                      <div key={i} className="bg-slate-800/50 rounded-lg p-4 text-center">
                        <Icon size={18} className={`mx-auto mb-2 ${m.color}`} />
                        <p className="text-2xl font-bold font-mono text-rail-300">{m.value}</p>
                        <p className="text-xs text-slate-400 mt-1">{m.label}</p>
                      </div>
                    )
                  })}
                </div>
              )
            ) : (
              <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                <Beaker size={32} className="mb-3" />
                <p>Configure parameters and run an evaluation</p>
                <p className="text-xs mt-1">Mean ± std over {nEpisodes} random episodes using median prototypes</p>
              </div>
            )}
          </Card>

          {result && !result.error && (
            <Card>
              <div className="flex items-center gap-2 mb-4">
                <TrendingUp size={18} className="text-rail-300" />
                <h2 className="font-semibold">Methodology</h2>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-rail-300 font-medium mb-1">Median Prototypes</p>
                  <p className="text-slate-400 text-xs">Class-median rather than mean embedding for robustness against outlier shots</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-rail-300 font-medium mb-1">Episodic Training</p>
                  <p className="text-slate-400 text-xs">N-way K-shot episodes with disjoint support/query splits</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3">
                  <p className="text-rail-300 font-medium mb-1">Open-Set Rejection</p>
                  <p className="text-slate-400 text-xs">Distance-to-prototype threshold prevents forced misclassification</p>
                </div>
              </div>
            </Card>
          )}
        </div>
      </div>
    </PageContainer>
  )
}
