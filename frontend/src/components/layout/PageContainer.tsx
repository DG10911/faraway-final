"use client"

import type { ReactNode } from "react"

interface PageContainerProps {
  title: string
  subtitle?: string
  eyebrow?: string
  children: ReactNode
  actions?: ReactNode
}

export function PageContainer({ title, subtitle, eyebrow, children, actions }: PageContainerProps) {
  return (
    <div className="relativez min-h-screen p-4 lg:p-8 max-w-[1400px] mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-7 animate-fade-up">
        <div>
          {eyebrow && <span className="eyebrow mb-2">{eyebrow}</span>}
          <h1 className="text-3xl font-bold tracking-tight gradient-text mt-1">{title}</h1>
          {subtitle && <p className="text-sm text-slate-500 mt-1.5 max-w-2xl">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
      {children}
    </div>
  )
}
