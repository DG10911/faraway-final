"use client"

import { AlertTriangle, RefreshCw } from "lucide-react"

export default function Error({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="glass rounded-xl p-8 max-w-md w-full text-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-red-900/40 flex items-center justify-center mx-auto">
          <AlertTriangle size={32} className="text-red-400" />
        </div>
        <h2 className="text-xl font-bold">Something went wrong</h2>
        <p className="text-sm text-gray-400">{error.message || "An unexpected error occurred."}</p>
        <button
          onClick={reset}
          className="inline-flex items-center gap-2 px-4 py-2 bg-rail-600 hover:bg-rail-500 rounded-lg text-sm font-medium transition-colors"
        >
          <RefreshCw size={16} />
          Try again
        </button>
      </div>
    </div>
  )
}
