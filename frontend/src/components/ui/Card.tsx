"use client"

import type { ReactNode } from "react"

interface CardProps {
  children: ReactNode
  className?: string
  glow?: "anomaly" | "healthy" | "none"
  onClick?: () => void
}

export function Card({ children, className = "", glow = "none", onClick }: CardProps) {
  const glowClass = glow === "anomaly" ? "anomaly-glow" : glow === "healthy" ? "healthy-glow" : ""
  const clickClass = onClick ? "cursor-pointer hover:border-rail-500/50 transition-all" : ""
  return (
    <div
      onClick={onClick}
      className={`glass rounded-xl p-6 ${glowClass} ${clickClass} ${className}`}
    >
      {children}
    </div>
  )
}
