"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { Menu, X, FlaskConical, Wifi, WifiOff } from "lucide-react"
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
    <nav className="sticky top-0 z-50 glass border-b border-gray-800/80">
      <div className="flex items-center justify-between h-14 px-4 lg:px-6">
        <div className="flex items-center gap-3">
          <button
            onClick={onToggleSidebar}
            className="lg:hidden p-2 -ml-2 rounded-lg hover:bg-gray-800/60 transition-colors"
            aria-label="Toggle sidebar"
          >
            {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
          <Link href="/" className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-rail-600 flex items-center justify-center">
              <FlaskConical size={18} className="text-white" />
            </div>
            <div>
              <span className="font-bold text-base tracking-tight">
                RailGuard<span className="text-rail-400">-FSL++</span>
              </span>
            </div>
          </Link>
        </div>

        <div className="flex items-center gap-4">
          <StatusDot
            status={apiStatus}
            label={apiStatus === "online" ? "API Ready" : apiStatus === "warning" ? "Not Initialized" : "API Offline"}
          />
          <div className="hidden sm:flex items-center gap-1 text-xs text-gray-500">
            {apiStatus === "online" ? (
              <Wifi size={14} className="text-green-400" />
            ) : (
              <WifiOff size={14} className="text-red-400" />
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}
