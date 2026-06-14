"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Menu, X, TrainFront } from "lucide-react"
import { api } from "@/lib/api"
import { StatusDot } from "@/components/ui/StatusDot"

interface NavbarProps {
  onToggleSidebar: () => void
  sidebarOpen: boolean
}

export function Navbar({ onToggleSidebar, sidebarOpen }: NavbarProps) {
  const [apiStatus, setApiStatus] = useState<"online" | "offline" | "warning">("offline")

  useEffect(() => {
    const check = () =>
      api
        .health()
        .then((h) => setApiStatus(h.initialized ? "online" : "warning"))
        .catch(() => setApiStatus("offline"))
    check()
    const interval = setInterval(check, 10000)
    return () => clearInterval(interval)
  }, [])

  return (
    <nav className="sticky top-0 z-50 glass-strong border-b border-slate-700">
      <div className="flex items-center justify-between h-16 px-4 lg:px-6">
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="lg:hidden p-2 -ml-2 rounded-lg hover:bg-slate-800 transition-colors"
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <Link href="/" className="flex items-center gap-3 group">
            <div className="w-9 h-9 rounded-xl bg-brand-gradient flex items-center justify-center shadow-sm group-hover:scale-105 transition-transform">
              <TrainFront size={18} className="text-white" />
            </div>
            <div className="leading-tight">
              <span className="font-bold text-base tracking-tight block text-slate-100">
                RailGuard<span className="text-rail-300">-FSL++</span>
              </span>
              <span className="text-[10px] uppercase tracking-[0.16em] text-slate-500">Few-Shot Rail AI</span>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-3">
          <Link href="/upload" className="hidden sm:inline-flex btn btn-ghost py-2 px-3 text-xs">
            Run a detection
          </Link>
          <StatusDot
            status={apiStatus}
            label={apiStatus === "online" ? "API Ready" : apiStatus === "warning" ? "Not Initialized" : "API Offline"}
          />
        </div>
      </div>
    </nav>
  )
}
