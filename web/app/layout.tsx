import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'AutoHDR Clone - AI Real Estate Photo Editing',
  description: 'Open-source HDR blending, sky replacement, day-to-dusk conversion',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-white min-h-screen">
        {children}
      </body>
    </html>
  )
}
