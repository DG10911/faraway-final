"use client"

interface StatusDotProps {
  status: "online" | "offline" | "warning"
  label?: string
  className?: string
}

const DOT_COLORS = {
  online: "bg-emerald-400",
  warning: "bg-gold-400",
  offline: "bg-danger-400",
}
const PILL_COLORS = {
  online: "text-emerald-300 border-emerald-500/30 bg-emerald-500/15",
  warning: "text-gold-500 border-gold-400/40 bg-gold-400/10",
  offline: "text-danger-400 border-danger-500/30 bg-danger-500/15",
}

export function StatusDot({ status, label, className = "" }: StatusDotProps) {
  return (
    <div className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 ${PILL_COLORS[status]} ${className}`}>
      <span className="relative flex h-2 w-2">
        <span className={`absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping ${DOT_COLORS[status]}`} />
        <span className={`relative inline-flex h-2 w-2 rounded-full ${DOT_COLORS[status]}`} />
      </span>
      {label && <span className="text-xs font-medium">{label}</span>}
    </div>
  )
}
