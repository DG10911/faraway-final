"use client"

import type { ReactNode } from "react"

interface PageContainerProps {
  title: string
  subtitle?: string
  children: ReactNode
  actions?: ReactNode
}

export function PageContainer({ title, subtitle, children, actions }: PageContainerProps) {
  return (
    <div className="min-h-screen p-4 lg:p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {subtitle && <p className="text-sm text-gray-400 mt-0.5">{subtitle}</p>}
        </div>
        {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
      </div>
      {children}
    </div>
  )
}
