"use client"

import { usePathname } from "next/navigation"
import Link from "next/link"
import {
  LayoutDashboard,
  Radio,
  Upload,
  Search,
  Beaker,
  Wrench,
  Eye,
} from "lucide-react"

const NAV_ITEMS = [
  { label: "Dashboard", href: "/", icon: LayoutDashboard },
  { label: "Live Monitoring", href: "/live", icon: Radio },
  { label: "Upload Detection", href: "/upload", icon: Upload },
  { label: "Unknown Discovery", href: "/discovery", icon: Search },
  { label: "Few-Shot Lab", href: "/few-shot", icon: Beaker },
  { label: "Maintenance", href: "/maintenance", icon: Wrench },
  { label: "Explainability", href: "/explain", icon: Eye },
]

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname()

  return (
    <>
      {open && (
        <div className="fixed inset-0 bg-black/50 z-40 lg:hidden" onClick={onClose} />
      )}
      <aside
        className={`fixed top-14 left-0 z-40 h-[calc(100vh-3.5rem)] w-60 glass border-r border-gray-800/80 transition-transform duration-200 lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <nav className="p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon
            const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onClose}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  isActive
                    ? "bg-rail-900/60 text-rail-300 border border-rail-800/50"
                    : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/40"
                }`}
              >
                <Icon size={18} className={isActive ? "text-rail-400" : ""} />
                {item.label}
              </Link>
            )
          })}
        </nav>

        <div className="absolute bottom-4 left-3 right-3 p-3 rounded-lg bg-gray-900/60 border border-gray-800">
          <p className="text-xs text-gray-500 leading-relaxed">
            <span className="text-rail-400 font-medium">RailGuard-FSL++</span>
            <br />
            v1.0.0 — Research Platform
          </p>
        </div>
      </aside>
    </>
  )
}
