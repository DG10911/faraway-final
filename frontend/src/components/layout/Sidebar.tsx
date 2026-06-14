"use client"

import { usePathname } from "next/navigation"
import Link from "next/link"
import { LayoutDashboard, Radio, Upload, Search, Beaker, Wrench, Eye, ShieldCheck } from "lucide-react"

const NAV_GROUPS: { title: string; items: { label: string; href: string; icon: any }[] }[] = [
  {
    title: "Overview",
    items: [
      { label: "Dashboard", href: "/", icon: LayoutDashboard },
      { label: "Live Monitoring", href: "/live", icon: Radio },
    ],
  },
  {
    title: "Detect",
    items: [
      { label: "Upload Detection", href: "/upload", icon: Upload },
      { label: "Unknown Discovery", href: "/discovery", icon: Search },
      { label: "Explainability", href: "/explain", icon: Eye },
    ],
  },
  {
    title: "Operate",
    items: [
      { label: "Few-Shot Lab", href: "/few-shot", icon: Beaker },
      { label: "Safety & Calibration", href: "/safety", icon: ShieldCheck },
      { label: "Maintenance", href: "/maintenance", icon: Wrench },
    ],
  },
]

interface SidebarProps {
  open: boolean
  onClose: () => void
}

export function Sidebar({ open, onClose }: SidebarProps) {
  const pathname = usePathname()

  return (
    <>
      {open && <div className="fixed inset-0 bg-slate-900/30 backdrop-blur-sm z-40 lg:hidden" onClick={onClose} />}
      <aside
        className={`fixed top-16 left-0 z-40 h-[calc(100vh-4rem)] w-60 glass-strong border-r border-slate-700 transition-transform duration-200 lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <nav className="p-3 space-y-5 overflow-y-auto h-[calc(100%-5.5rem)]">
          {NAV_GROUPS.map((group) => (
            <div key={group.title}>
              <p className="px-3 mb-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
                {group.title}
              </p>
              <div className="space-y-1">
                {group.items.map((item) => {
                  const Icon = item.icon
                  const isActive = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href))
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      onClick={onClose}
                      className={`sidebar-link ${isActive ? "sidebar-link-active" : "sidebar-link-inactive"}`}
                    >
                      {isActive && <span className="absolute left-0 top-1/2 -translate-y-1/2 h-5 w-1 rounded-full bg-brand-gradient" />}
                      <Icon size={18} className={isActive ? "text-rail-300" : "text-slate-500"} />
                      {item.label}
                    </Link>
                  )
                })}
              </div>
            </div>
          ))}
        </nav>

        <div className="absolute bottom-3 left-3 right-3 p-3 rounded-xl stat-grad border border-slate-700">
          <p className="text-[11px] text-slate-400 leading-relaxed">
            <span className="text-rail-300 font-semibold">RailGuard-FSL++</span>
            <br />
            <span className="text-slate-500">v1.0 · Research Preview</span>
          </p>
        </div>
      </aside>
    </>
  )
}
