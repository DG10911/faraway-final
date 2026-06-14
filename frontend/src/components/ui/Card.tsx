"use client"

import type { ReactNode } from "react"

interface CardProps {
  children: ReactNode
  className?: string
  glow?: "anomaly" | "healthy" | "none"
  hover?: boolean
  onClick?: () => void
}

export function Card({ children, className = "", glow = "none", hover, onClick }: CardProps) {
  const glowClass = glow === "anomaly" ? "anomaly-glow" : glow === "healthy" ? "healthy-glow" : ""
  const hoverClass = hover || onClick ? "card-hover cursor-pointer" : ""
  return (
    <div onClick={onClick} className={`card-surface p-6 ${glowClass} ${hoverClass} ${className}`}>
      {children}
    </div>
  )
}
