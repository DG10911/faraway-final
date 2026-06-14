"use client"

import type { ReactNode } from "react"
import { SEVERITY_COLORS } from "@/lib/types"

interface BadgeProps {
  variant: keyof typeof SEVERITY_COLORS | "default" | "success" | "warning" | "error" | "info"
  children: ReactNode
  className?: string
}

const BADGE_COLORS: Record<string, string> = {
  ...SEVERITY_COLORS,
  default: "text-slate-300 bg-slate-800 border-slate-700",
  success: "text-emerald-300 bg-emerald-500/15 border-emerald-500/30",
  warning: "text-amber-400 bg-amber-500/15 border-amber-500/30",
  error: "text-red-400 bg-red-500/15 border-red-500/30",
  info: "text-rail-300 bg-rail-500/15 border-rail-500/30",
}

export function Badge({ variant, children, className = "" }: BadgeProps) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${BADGE_COLORS[variant] || BADGE_COLORS.default} ${className}`}>
      {children}
    </span>
  )
}
