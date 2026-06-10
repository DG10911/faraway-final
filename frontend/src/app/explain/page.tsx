"use client";

import { useState } from "react";

export default function ExplainabilityPage() {
  const [selectedView, setSelectedView] = useState<"gradcam" | "attention" | "heatmap">("heatmap");

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">🔬 Explainability Viewer</h1>
        <p className="text-gray-400 text-sm">Visualizing why defects were detected</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">Visualization Type</h2>
          <div className="space-y-2">
            {[
              { key: "heatmap", label: "Anomaly Heatmap", desc: "PatchCore anomaly scores per region" },
              { key: "gradcam", label: "Grad-CAM", desc: "Gradient-weighted class activation" },
              { key: "attention", label: "Attention Maps", desc: "DINOv2 self-attention visualization" },
            ].map((v) => (
              <button
                key={v.key}
                onClick={() => setSelectedView(v.key as any)}
                className={`w-full text-left p-3 rounded-lg transition-colors ${
                  selectedView === v.key ? "bg-rail-900 border border-rail-700" : "bg-gray-900 hover:bg-gray-800"
                }`}
              >
                <p className="font-medium text-sm">{v.label}</p>
                <p className="text-xs text-gray-500 mt-1">{v.desc}</p>
              </button>
            ))}
          </div>

          <div className="mt-6">
            <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-3">Sample Image</h3>
            <div className="aspect-square bg-gray-900 rounded-lg border border-gray-800 flex items-center justify-center">
              <span className="text-6xl">🛤️</span>
            </div>
          </div>
        </div>

        <div className="lg:col-span-2 glass rounded-xl p-6">
          <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
            {selectedView === "heatmap" && "Anomaly Heatmap"}
            {selectedView === "gradcam" && "Grad-CAM Visualization"}
            {selectedView === "attention" && "Attention Map"}
          </h2>

          <div className="aspect-video bg-gray-900 rounded-lg border border-gray-800 flex items-center justify-center">
            <div className="text-center">
              <span className="text-6xl block mb-4">
                {selectedView === "heatmap" && "🔥"}
                {selectedView === "gradcam" && "🎯"}
                {selectedView === "attention" && "👁️"}
              </span>
              <p className="text-gray-500">Upload an image or use detection to generate visualization</p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="bg-gray-900 rounded-lg p-4">
              <span className="text-xs text-gray-500">Detection Rationale</span>
              <p className="text-sm mt-1">
                {selectedView === "heatmap" && "Regions with high anomaly scores (red) indicate deviations from healthy rail memory bank."}
                {selectedView === "gradcam" && "Grad-CAM highlights pixels that most influenced the classifier's decision."}
                {selectedView === "attention" && "DINOv2 self-attention maps show which patch relationships the model focuses on."}
              </p>
            </div>
            <div className="bg-gray-900 rounded-lg p-4">
              <span className="text-xs text-gray-500">Research Value</span>
              <p className="text-sm mt-1">
                Enables engineers to verify that the model focuses on the rail surface, not background or lighting changes.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
