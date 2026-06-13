import type { Metadata, Viewport } from "next"
import { Inter } from "next/font/google"
import { ClientLayout } from "./client-layout"
import "./globals.css"

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" })

export const metadata: Metadata = {
  title: "RailGuard-FSL++ | Research Platform",
  description:
    "Few-Shot Defect Detection in Rail Infrastructure — Self-supervised representation learning, few-shot classification, open-set defect discovery, and predictive maintenance intelligence.",
  keywords: ["railway", "defect detection", "few-shot learning", "DINOv2", "PatchCore", "computer vision"],
  authors: [{ name: "RailGuard-FSL++ Team" }],
}

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#c84a6a",
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="bg-[#0f0a10] text-gray-100 min-h-screen font-sans antialiased">
        <ClientLayout>{children}</ClientLayout>
      </body>
    </html>
  )
}
