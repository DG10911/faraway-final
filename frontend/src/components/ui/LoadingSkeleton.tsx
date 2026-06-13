"use client"

interface SkeletonProps {
  className?: string
  variant?: "text" | "card" | "chart" | "image"
}

function SkeletonBlock({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-gray-800/60 ${className}`} />
}

export function LoadingSkeleton({ variant = "card", className = "" }: SkeletonProps) {
  if (variant === "text") {
    return (
      <div className={`space-y-3 ${className}`}>
        <SkeletonBlock className="h-4 w-3/4" />
        <SkeletonBlock className="h-4 w-1/2" />
        <SkeletonBlock className="h-4 w-5/6" />
      </div>
    )
  }

  if (variant === "chart") {
    return (
      <div className={`space-y-3 ${className}`}>
        <SkeletonBlock className="h-5 w-1/3" />
        <SkeletonBlock className="h-48 w-full" />
      </div>
    )
  }

  if (variant === "image") {
    return (
      <div className={`space-y-3 ${className}`}>
        <SkeletonBlock className="aspect-video w-full" />
        <SkeletonBlock className="h-4 w-2/3" />
      </div>
    )
  }

  return (
    <div className={`glass rounded-xl p-6 space-y-4 ${className}`}>
      <div className="flex items-center gap-3">
        <SkeletonBlock className="h-10 w-10 rounded-full" />
        <div className="space-y-2 flex-1">
          <SkeletonBlock className="h-4 w-1/3" />
          <SkeletonBlock className="h-3 w-1/4" />
        </div>
      </div>
      <SkeletonBlock className="h-24 w-full" />
      <div className="grid grid-cols-3 gap-3">
        <SkeletonBlock className="h-16" />
        <SkeletonBlock className="h-16" />
        <SkeletonBlock className="h-16" />
      </div>
    </div>
  )
}

export function DashboardSkeleton() {
  return (
    <div className="min-h-screen p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="space-y-2">
          <SkeletonBlock className="h-8 w-64" />
          <SkeletonBlock className="h-4 w-48" />
        </div>
        <SkeletonBlock className="h-4 w-24" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {Array.from({ length: 6 }).map((_, i) => (
          <LoadingSkeleton key={i} variant="card" />
        ))}
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LoadingSkeleton variant="chart" />
        <LoadingSkeleton variant="card" />
      </div>
    </div>
  )
}
