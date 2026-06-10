"use client";

import { useState } from "react";

export default function MaintenancePage() {
  const [tracks] = useState([
    { id: "Track A12", defects: 4, topRisk: 82, topDefect: "Crack", topSeverity: "High" },
    { id: "Track B7", defects: 2, topRisk: 67, topDefect: "Squat", topSeverity: "Medium" },
    { id: "Track C3", defects: 3, topRisk: 91, topDefect: "Unknown Anomaly", topSeverity: "Critical" },
    { id: "Track D9", defects: 1, topRisk: 34, topDefect: "Spalling", topSeverity: "Low" },
  ]);

  const [priorities] = useState([
    { rank: "#1", track: "Track C3", type: "Unknown Anomaly", risk: 91, severity: "Critical" },
    { rank: "#2", track: "Track A12", type: "Crack", risk: 82, severity: "High" },
    { rank: "#3", track: "Track B7", type: "Squat", risk: 67, severity: "Medium" },
    { rank: "#4", track: "Track D9", type: "Spalling", risk: 34, severity: "Low" },
  ]);

  const severityColor: Record<string, string> = {
    Low: "text-green-400",
    Medium: "text-yellow-400",
    High: "text-orange-400",
    Critical: "text-danger-400",
  };

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">⚙️ Maintenance Priorities</h1>
        <p className="text-gray-400 text-sm">Digital twin predictive maintenance intelligence</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Railway Segments</h2>
          <div className="space-y-3">
            {tracks.map((t) => (
              <div key={t.id} className="bg-gray-900 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{t.id}</span>
                  <span className={`text-sm ${severityColor[t.topSeverity]}`}>{t.topSeverity}</span>
                </div>
                <div className="mt-2 flex gap-4 text-sm">
                  <span className="text-gray-400">{t.defects} active defects</span>
                  <span className="text-gray-400">Top: {t.topDefect}</span>
                  <span className="text-danger-400 font-mono">{t.topRisk}% risk</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Priority Rankings</h2>
          <div className="space-y-3">
            {priorities.map((p) => (
              <div key={p.rank} className="flex items-center justify-between bg-gray-900 rounded-lg p-4">
                <div className="flex items-center gap-3">
                  <span className="w-8 h-8 rounded-full bg-rail-900 text-rail-300 flex items-center justify-center text-sm font-bold">
                    {p.rank}
                  </span>
                  <div>
                    <p className="font-medium">{p.track}</p>
                    <p className="text-xs text-gray-400">{p.type}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-bold ${severityColor[p.severity]}`}>{p.severity}</p>
                  <p className="text-sm text-danger-400 font-mono">{p.risk}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="glass rounded-xl p-6">
        <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Digital Twin Logic</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
          <div className="bg-gray-900 rounded-lg p-3">
            <span className="text-gray-500">Failure Risk</span>
            <p className="font-mono mt-1">anomaly_score × 0.6 + severity_weight × 0.4</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <span className="text-gray-500">Severity Weights</span>
            <p className="font-mono mt-1">Low=0.2 Med=0.4 High=0.7 Crit=0.95</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <span className="text-gray-500">Priority</span>
            <p className="font-mono mt-1">Sorted by failure risk desc</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-3">
            <span className="text-gray-500">Health Score</span>
            <p className="font-mono mt-1">1.0 - avg_failure_risk</p>
          </div>
        </div>
      </div>
    </div>
  );
}
