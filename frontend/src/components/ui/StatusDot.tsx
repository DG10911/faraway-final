"use client"

interface StatusDotProps {
  status: "online" | "offline" | "warning"
  label?: string
  className?: string
}

const DOT_COLORS = {
  online: "bg-green-400 shadow-[0_0_8px_rgba(74,222,128,0.5)]",
  warning: "bg-yellow-400 shadow-[0_0_8px_rgba(250,204,21,0.5)]",
  offline: "bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.5)]",
}

export function StatusDot({ status, label, className = "" }: StatusDotProps) {
  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <span className={`w-2 h-2 rounded-full ${DOT_COLORS[status]} animate-pulse`} />
      {label && <span className="text-sm text-gray-400">{label}</span>}
    </div>
  )
}
