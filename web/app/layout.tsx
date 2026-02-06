import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'HDR it - AI Real Estate Photo Editing',
  description: 'Professional HDR blending, sky replacement, day-to-dusk conversion for real estate photography',
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
