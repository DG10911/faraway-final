"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";

export default function UploadDetection() {
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setPreview(URL.createObjectURL(file));
    setLoading(true);
    setResult(null);

    const form = new FormData();
    form.append("file", file);
    form.append("track_id", "Upload_Track");
    form.append("location_m", "0");

    try {
      const res = await fetch("http://localhost:8000/detect", { method: "POST", body: form });
      const data = await res.json();
      setResult(data);
    } catch {
      setResult({ status: "error", reason: "API unreachable" });
    } finally {
      setLoading(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept: { "image/*": [".png", ".jpg", ".jpeg"] }, maxFiles: 1 });

  const severityColor: Record<string, string> = {
    Low: "text-green-400",
    Medium: "text-yellow-400",
    High: "text-orange-400",
    Critical: "text-danger-400",
  };

  return (
    <div className="min-h-screen p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold">📤 Upload Detection</h1>
        <p className="text-gray-400 text-sm">Analyze a single rail image for defects</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div
          {...getRootProps()}
          className={`glass rounded-xl p-12 text-center cursor-pointer border-2 border-dashed transition-all ${
            isDragActive ? "border-rail-400 bg-rail-900/20" : "border-gray-700 hover:border-gray-500"
          }`}
        >
          <input {...getInputProps()} />
          <span className="text-5xl block mb-4">📷</span>
          {isDragActive ? (
            <p className="text-rail-400">Drop image here...</p>
          ) : (
            <div>
              <p className="text-gray-300">Drag & drop a rail image, or click to select</p>
              <p className="text-xs text-gray-500 mt-2">PNG, JPG up to 16MB</p>
            </div>
          )}
        </div>

        <div className="glass rounded-xl p-6">
          {loading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="animate-spin text-4xl mb-2">⚙️</div>
                <p className="text-gray-400">Analyzing frame...</p>
              </div>
            </div>
          )}

          {!loading && !result && preview && (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">Awaiting analysis...</p>
            </div>
          )}

          {!loading && !preview && (
            <div className="flex items-center justify-center h-full">
              <p className="text-gray-500">Upload an image to begin</p>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <span
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    result.status === "defect_detected"
                      ? "bg-danger-900/50 text-danger-400"
                      : result.status === "healthy"
                      ? "bg-green-900/50 text-green-400"
                      : "bg-yellow-900/50 text-yellow-400"
                  }`}
                >
                  {result.status}
                </span>
                {result.reason && <span className="text-sm text-gray-400">{result.reason}</span>}
              </div>

              {result.label && (
                <div>
                  <span className="text-gray-400 text-sm">Defect Type</span>
                  <p className="text-xl font-bold">{result.label.replace("_", " ")}</p>
                </div>
              )}

              {result.confidence && (
                <div>
                  <span className="text-gray-400 text-sm">Confidence</span>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-gray-800 rounded-full">
                      <div
                        className="h-full bg-rail-400 rounded-full"
                        style={{ width: `${(result.confidence * 100).toFixed(0)}%` }}
                      />
                    </div>
                    <span className="text-sm font-mono">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              )}

              {result.anomaly_score !== undefined && (
                <div>
                  <span className="text-gray-400 text-sm">Anomaly Score</span>
                  <p className="text-lg font-mono">{result.anomaly_score.toFixed(4)}</p>
                </div>
              )}

              {result.severity && (
                <div>
                  <span className="text-gray-400 text-sm">Severity</span>
                  <p className={`text-lg font-bold ${severityColor[result.severity.severity] || "text-gray-300"}`}>
                    {result.severity.severity}
                  </p>
                </div>
              )}

              {result.failure_risk !== undefined && (
                <div>
                  <span className="text-gray-400 text-sm">Failure Risk</span>
                  <p className="text-lg font-mono text-danger-400">{result.failure_risk}%</p>
                </div>
              )}

              {result.heatmap_b64 && (
                <div>
                  <span className="text-gray-400 text-sm">Anomaly Heatmap</span>
                  <img
                    src={`data:image/png;base64,${result.heatmap_b64}`}
                    alt="Heatmap"
                    className="mt-1 rounded-lg border border-gray-800"
                  />
                </div>
              )}

              {result.gradcam_b64 && (
                <div>
                  <span className="text-gray-400 text-sm">Grad-CAM</span>
                  <img
                    src={`data:image/png;base64,${result.gradcam_b64}`}
                    alt="Grad-CAM"
                    className="mt-1 rounded-lg border border-gray-800"
                  />
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
