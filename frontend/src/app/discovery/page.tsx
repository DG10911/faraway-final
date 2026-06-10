"use client";

import { useState } from "react";

export default function DiscoveryPage() {
  const [samples] = useState([
    { id: 1, label: "Unknown #001", confidence: 0.93, timestamp: "2026-06-10 14:32", track: "A12" },
    { id: 2, label: "Unknown #002", confidence: 0.87, timestamp: "2026-06-10 13:15", track: "B7" },
    { id: 3, label: "Unknown #003", confidence: 0.71, timestamp: "2026-06-10 11:44", track: "C3" },
    { id: 4, label: "Unknown #004", confidence: 0.95, timestamp: "2026-06-10 10:02", track: "A12" },
  ]);

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">🔍 Unknown Defect Discovery</h1>
        <p className="text-gray-400 text-sm">Open-set detections that don't match known defect prototypes</p>
      </header>

      <div className="glass rounded-xl p-6 mb-6">
        <h2 className="text-lg font-semibold mb-2">Research Insight</h2>
        <p className="text-gray-400 text-sm">
          When distance_to_nearest_prototype &gt; threshold AND anomaly_score &gt; threshold,
          the system returns <code className="text-rail-400">unknown_anomaly</code> instead of forcing
          assignment to known classes. This enables discovery of novel defect categories.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {samples.map((s) => (
          <div key={s.id} className="glass rounded-xl p-5 border border-yellow-900/50">
            <div className="flex items-center justify-between mb-3">
              <span className="font-mono text-yellow-400">{s.label}</span>
              <span className="text-xs text-gray-500">{s.timestamp}</span>
            </div>
            <div className="flex gap-4 text-sm">
              <div>
                <span className="text-gray-500">Confidence</span>
                <p className="font-mono">{(s.confidence * 100).toFixed(0)}%</p>
              </div>
              <div>
                <span className="text-gray-500">Track</span>
                <p className="font-mono">{s.track}</p>
              </div>
            </div>
            <div className="mt-3 flex gap-2">
              <button className="px-3 py-1 text-xs bg-rail-900 text-rail-300 rounded-lg hover:bg-rail-800">
                Add to Few-Shot Lab
              </button>
              <button className="px-3 py-1 text-xs bg-gray-800 text-gray-400 rounded-lg hover:bg-gray-700">
                View Details
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
