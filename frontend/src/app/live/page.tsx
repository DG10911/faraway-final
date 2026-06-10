"use client";

import { useState, useEffect } from "react";

export default function LiveMonitoring() {
  const [frame, setFrame] = useState(0);
  const [status, setStatus] = useState<string | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setFrame((f) => f + 1);
      fetch("http://localhost:8000/health")
        .then((r) => r.json())
        .then((d) => setStatus(d.initialized ? "Model Ready" : "Standby"))
        .catch(() => setStatus("Disconnected"));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">📡 Live Monitoring</h1>
        <p className="text-gray-400 text-sm">Real-time rail inspection feed</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 glass rounded-xl p-4">
          <div className="aspect-video bg-gray-900 rounded-lg flex items-center justify-center border border-gray-800">
            <div className="text-center">
              <span className="text-4xl">🚂</span>
              <p className="text-gray-500 mt-2">Live feed placeholder</p>
              <p className="text-xs text-gray-600">Frame #{frame}</p>
            </div>
          </div>
          {status && (
            <div className={`mt-2 text-sm ${status === "Model Ready" ? "text-green-400" : "text-yellow-400"}`}>
              {status}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">System Status</h3>
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">FPS</span>
                <span className="font-mono text-green-400">30.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Latency</span>
                <span className="font-mono">42ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Frames Processed</span>
                <span className="font-mono">{frame * 60}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Anomalies Found</span>
                <span className="font-mono text-danger-400">3</span>
              </div>
            </div>
          </div>

          <div className="glass rounded-xl p-4">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">Recent Alerts</h3>
            <div className="mt-3 space-y-2">
              {["Crack detected on Track A12", "Unknown anomaly at 342m", "Squat confirmed on Track B7"].map(
                (alert, i) => (
                  <div key={i} className="flex items-center gap-2 text-sm p-2 bg-gray-900 rounded">
                    <span className="w-2 h-2 rounded-full bg-danger-500" />
                    <span>{alert}</span>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
