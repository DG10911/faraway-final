"use client";

import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import Link from "next/link";

const navItems = [
  { label: "Live Monitoring", href: "/live", icon: "📡" },
  { label: "Upload Detection", href: "/upload", icon: "📤" },
  { label: "Unknown Defect Discovery", href: "/discovery", icon: "🔍" },
  { label: "Few-Shot Learning Lab", href: "/few-shot", icon: "🧪" },
  { label: "Maintenance Priorities", href: "/maintenance", icon: "⚙️" },
  { label: "Explainability Viewer", href: "/explain", icon: "🔬" },
];

const placeholderData = [
  { name: "Crack", count: 12 },
  { name: "Squat", count: 8 },
  { name: "Spalling", count: 5 },
  { name: "Flaking", count: 3 },
  { name: "Shelling", count: 2 },
  { name: "Unknown", count: 4 },
];

export default function Home() {
  const [status, setStatus] = useState("connecting...");

  useEffect(() => {
    fetch("http://localhost:8000/health")
      .then((r) => r.json())
      .then((d) => setStatus(d.initialized ? "Model Ready" : "Not Initialized"))
      .catch(() => setStatus("API Offline"));
  }, []);

  return (
    <div className="min-h-screen p-6">
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">
              RailGuard<span className="text-rail-400">-FSL++</span>
            </h1>
            <p className="text-gray-400 mt-1">Few-Shot Defect Detection in Rail Infrastructure</p>
          </div>
          <div className="flex items-center gap-3">
            <span className={`w-2 h-2 rounded-full ${status === "Model Ready" ? "bg-green-400" : "bg-yellow-400"}`} />
            <span className="text-sm text-gray-400">{status}</span>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        {navItems.map((item) => (
          <Link
            key={item.href}
            href={item.href}
            className="glass rounded-xl p-5 hover:border-rail-500/50 transition-all group"
          >
            <span className="text-2xl mb-2 block">{item.icon}</span>
            <span className="text-lg font-medium group-hover:text-rail-400 transition-colors">{item.label}</span>
          </Link>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Defect Distribution (Last 24h)</h2>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={placeholderData}>
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
        </div>

        <div className="glass rounded-xl p-6">
          <h2 className="text-lg font-semibold mb-4">Research Pipeline</h2>
          <div className="space-y-3 text-sm">
            {[
              "Self-Supervised Representation Learning (DINOv2)",
              "Healthy-Only Anomaly Detection (PatchCore)",
              "Few-Shot Classification (Prototypical Networks)",
              "Open-Set Unknown Defect Discovery",
              "Cross-Domain Adaptation (Mendeley → RSDDs)",
              "Predictive Maintenance Digital Twin",
            ].map((step, i) => (
              <div key={i} className="flex items-center gap-3">
                <span className="w-6 h-6 rounded-full bg-rail-900 text-rail-300 flex items-center justify-center text-xs font-bold">
                  {i + 1}
                </span>
                <span className="text-gray-300">{step}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
