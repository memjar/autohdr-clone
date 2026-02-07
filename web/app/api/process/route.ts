import { NextRequest, NextResponse } from 'next/server'

// Vercel Pro allows 5 minute function timeout
export const maxDuration = 300 // 5 minutes

const BACKEND_URL = process.env.BACKEND_URL || 'https://hdr.it.com.ngrok.pro'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const searchParams = request.nextUrl.searchParams
    
    // Forward to backend
    const backendUrl = `${BACKEND_URL}/process?${searchParams.toString()}`
    
    const response = await fetch(backendUrl, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      const error = await response.text()
      return NextResponse.json({ error }, { status: response.status })
    }

    // Stream the image back
    const blob = await response.blob()
    return new NextResponse(blob, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Content-Disposition': `attachment; filename="hdrit_${Date.now()}.jpg"`,
      },
    })
  } catch (error: any) {
    console.error('Proxy error:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
