import Link from "next/link"
import { Home } from "lucide-react"

export default function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="card-surface p-8 max-w-md w-full text-center space-y-4">
        <div className="text-6xl">🛤️</div>
        <h2 className="text-xl font-bold">Page not found</h2>
        <p className="text-sm text-slate-400">This rail segment doesn&apos;t exist on our map.</p>
        <Link
          href="/"
          className="inline-flex items-center gap-2 px-4 py-2 bg-rail-600 hover:bg-rail-700 text-white rounded-lg text-sm font-medium transition-colors"
        >
          <Home size={16} />
          Back to Dashboard
        </Link>
      </div>
    </div>
  )
}
