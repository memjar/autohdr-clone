import type { Metadata, Viewport } from 'next'
import { ClerkProvider } from '@clerk/nextjs'
import './globals.css'

// Check if Clerk keys are configured
const clerkPubKey = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
const isClerkConfigured = clerkPubKey && clerkPubKey.startsWith('pk_')

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5, // Allow zoom for accessibility
  userScalable: true, // Accessibility requirement
  viewportFit: 'cover', // Support notch/Dynamic Island
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#0a0a0f' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0f' },
  ],
}

export const metadata: Metadata = {
  title: 'HDRit - AI Real Estate Photo Editing',
  description: 'Professional HDR blending, sky replacement, day-to-dusk conversion for real estate photography. Made by Virul.',
  keywords: ['HDR', 'real estate photography', 'photo editing', 'AI', 'twilight', 'sky replacement'],
  authors: [{ name: 'Virul', url: 'https://virul.co' }],
  creator: 'Virul',
  publisher: 'HDRit',
  formatDetection: {
    telephone: false, // Prevent auto-linking phone numbers
    email: false,
    address: false,
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'HDRit',
  },
  openGraph: {
    title: 'HDRit - AI Real Estate Photo Editing',
    description: 'Professional HDR blending, sky replacement, day-to-dusk conversion for real estate photography.',
    type: 'website',
    locale: 'en_US',
    siteName: 'HDRit',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'HDRit - AI Real Estate Photo Editing',
    description: 'Professional HDR blending, sky replacement, day-to-dusk conversion.',
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  const content = (
    <html lang="en" className="antialiased">
      <head>
        {/* Preconnect to backend for faster API calls */}
        <link rel="preconnect" href="https://hdr.it.com.ngrok.pro" />
        <link rel="dns-prefetch" href="https://hdr.it.com.ngrok.pro" />
        {/* PWA manifest */}
        <link rel="manifest" href="/manifest.json" />
        {/* Favicon */}
        <link rel="icon" href="/favicon.ico" sizes="any" />
      </head>
      <body className="bg-[#0a0a0f] text-white min-h-screen">
        {children}
      </body>
    </html>
  )

  // Only wrap with ClerkProvider if keys are configured
  if (isClerkConfigured) {
    return (
      <ClerkProvider
        appearance={{
          variables: {
            colorPrimary: '#3b82f6',
            colorBackground: '#0a0a0f',
            colorText: '#ffffff',
            colorTextSecondary: '#9ca3af',
            colorInputBackground: '#1f2937',
            colorInputText: '#ffffff',
          },
        }}
      >
        {content}
      </ClerkProvider>
    )
  }

  return content
}
