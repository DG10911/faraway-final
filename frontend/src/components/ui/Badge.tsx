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
  default: "text-gray-400 bg-gray-800 border-gray-700",
  success: "text-green-400 bg-green-900/40 border-green-700/40",
  warning: "text-yellow-400 bg-yellow-900/40 border-yellow-700/40",
  error: "text-red-400 bg-red-900/40 border-red-700/40",
  info: "text-sky-400 bg-sky-900/40 border-sky-700/40",
}

export function Badge({ variant, children, className = "" }: BadgeProps) {
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium border ${BADGE_COLORS[variant] || BADGE_COLORS.default} ${className}`}>
      {children}
    </span>
  )
}
