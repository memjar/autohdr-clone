import { NextRequest, NextResponse } from 'next/server'

// Vercel Pro allows 5 minute function timeout
export const maxDuration = 300 // 5 minutes

const BACKEND_URL = process.env.BACKEND_URL || 'https://hdr.it.com.ngrok.pro'

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const searchParams = request.nextUrl.searchParams

    // Reconstruct FormData for forwarding
    const forwardFormData = new FormData()
    for (const [key, value] of formData.entries()) {
      forwardFormData.append(key, value)
    }

    // Forward to backend
    const backendUrl = `${BACKEND_URL}/process?${searchParams.toString()}`
    console.log('Forwarding to:', backendUrl)

    const response = await fetch(backendUrl, {
      method: 'POST',
      body: forwardFormData,
    })

    console.log('Backend response:', response.status)

    if (!response.ok) {
      const error = await response.text()
      console.error('Backend error:', error)
      return NextResponse.json({ error }, { status: response.status })
    }

    // Get the image blob
    const imageBuffer = await response.arrayBuffer()
    console.log('Received image:', imageBuffer.byteLength, 'bytes')

    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'image/jpeg',
        'Content-Length': imageBuffer.byteLength.toString(),
        'Content-Disposition': `attachment; filename="hdrit_${Date.now()}.jpg"`,
      },
    })
  } catch (error: any) {
    console.error('Proxy error:', error)
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
