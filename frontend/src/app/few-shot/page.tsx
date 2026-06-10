"use client";

import { useState } from "react";

export default function FewShotLab() {
  const [nShots, setNShots] = useState(5);
  const [nWays, setNWays] = useState(5);
  const [nEpisodes, setNEpisodes] = useState(100);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const runEval = async () => {
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/evaluate/few-shot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_ways: nWays, n_shots: nShots, n_episodes: nEpisodes }),
      });
      const data = await res.json();
      setResult(data);
    } catch {
      setResult({ error: "API unreachable" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">🧪 Few-Shot Learning Lab</h1>
        <p className="text-gray-400 text-sm">Episodic training with median prototypes</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Configuration</h2>
          <div className="space-y-4">
            <div>
              <label className="text-sm text-gray-400">N-Shot</label>
              <select
                value={nShots}
                onChange={(e) => setNShots(Number(e.target.value))}
                className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
              >
                {[1, 3, 5, 10].map((n) => (
                  <option key={n} value={n}>
                    {n}-shot
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-sm text-gray-400">N-Way</label>
              <select
                value={nWays}
                onChange={(e) => setNWays(Number(e.target.value))}
                className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
              >
                {[2, 3, 5, 7].map((n) => (
                  <option key={n} value={n}>
                    {n}-way
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="text-sm text-gray-400">Episodes</label>
              <input
                type="number"
                value={nEpisodes}
                onChange={(e) => setNEpisodes(Number(e.target.value))}
                className="w-full mt-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-sm"
              />
            </div>

            <button
              onClick={runEval}
              disabled={loading}
              className="w-full py-2 bg-rail-600 hover:bg-rail-500 rounded-lg font-medium transition-colors disabled:opacity-50"
            >
              {loading ? "Running..." : "Run Evaluation"}
            </button>
          </div>
        </div>

        <div className="lg:col-span-2 glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Results</h2>

          {result && !result.error ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: "Mean Accuracy", value: `${(result.mean_accuracy * 100).toFixed(1)}%` },
                  { label: "Std Dev", value: `${(result.std_accuracy * 100).toFixed(1)}%` },
                  { label: "Episodes", value: result.n_episodes },
                  { label: "Open-Set Detection", value: result.open_set_detection_rate !== undefined ? `${(result.open_set_detection_rate * 100).toFixed(1)}%` : "N/A" },
                ].map((m, i) => (
                  <div key={i} className="bg-gray-900 rounded-lg p-4 text-center">
                    <p className="text-xs text-gray-400">{m.label}</p>
                    <p className="text-xl font-bold text-rail-400 mt-1">{m.value}</p>
                  </div>
                ))}
              </div>
              <p className="text-sm text-gray-500">
                Results are mean ± std over {result.n_episodes} random episodes using median prototypes.
              </p>
            </div>
          ) : (
            <div className="flex items-center justify-center h-48 text-gray-500">
              {result?.error ? `Error: ${result.error}` : "Configure and run an evaluation to see results"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
