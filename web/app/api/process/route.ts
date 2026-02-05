import { NextRequest, NextResponse } from 'next/server'
import sharp from 'sharp'

export const runtime = 'nodejs'
export const maxDuration = 60

/**
 * Process uploaded images with HDR-like enhancement
 *
 * POST /api/process
 * Body: FormData with 'images' field (1 or more files)
 * Query: ?mode=hdr|twilight
 *
 * Returns: Processed JPEG image
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const mode = request.nextUrl.searchParams.get('mode') || 'hdr'

    // Get all uploaded images
    const imageFiles: File[] = []
    for (const [key, value] of formData.entries()) {
      if (key === 'images' && value instanceof File) {
        imageFiles.push(value)
      }
    }

    if (imageFiles.length === 0) {
      return NextResponse.json(
        { error: 'No images provided' },
        { status: 400 }
      )
    }

    // Convert files to buffers
    const buffers = await Promise.all(
      imageFiles.map(async (file) => {
        const arrayBuffer = await file.arrayBuffer()
        return Buffer.from(arrayBuffer)
      })
    )

    let processedBuffer: Buffer

    if (mode === 'twilight') {
      // Day-to-dusk processing (single image)
      processedBuffer = await processTwilight(buffers[0])
    } else {
      // HDR merge processing
      processedBuffer = await processHDR(buffers)
    }

    // Return processed image
    return new NextResponse(processedBuffer, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Content-Disposition': `attachment; filename="processed_${Date.now()}.jpg"`,
      },
    })
  } catch (error: any) {
    console.error('Processing error:', error)
    return NextResponse.json(
      { error: error.message || 'Processing failed' },
      { status: 500 }
    )
  }
}

/**
 * HDR-like processing for bracketed images
 * Applies exposure blending simulation and enhancement
 */
async function processHDR(buffers: Buffer[]): Promise<Buffer> {
  // For single image, just enhance it
  if (buffers.length === 1) {
    return enhanceImage(buffers[0])
  }

  // For multiple brackets, use the middle one as base and enhance
  // (Full HDR merge would require more complex processing)
  const middleIndex = Math.floor(buffers.length / 2)
  const baseBuffer = buffers[middleIndex]

  // Get image metadata
  const metadata = await sharp(baseBuffer).metadata()

  // Process the base image with HDR-like enhancement
  const enhanced = await sharp(baseBuffer)
    // Normalize and enhance
    .normalize()
    // Adjust levels - lift shadows, control highlights
    .modulate({
      brightness: 1.05,  // Slight brightness boost
      saturation: 1.1,   // +10% saturation (matches target spec)
    })
    // Apply contrast curve (simulates S-curve)
    .linear(1.1, -10) // Slight contrast boost
    // Sharpen subtly
    .sharpen({
      sigma: 0.8,
      m1: 0.5,
      m2: 0.5,
    })
    // Output as high-quality JPEG
    .jpeg({
      quality: 85,
      chromaSubsampling: '4:4:4',
    })
    .toBuffer()

  return enhanced
}

/**
 * Day-to-dusk twilight conversion
 */
async function processTwilight(buffer: Buffer): Promise<Buffer> {
  const processed = await sharp(buffer)
    // Reduce brightness for dusk effect
    .modulate({
      brightness: 0.75,
      saturation: 1.15,
    })
    // Shift toward cooler/blue tones
    .tint({ r: 180, g: 190, b: 220 })
    // Add warmth to simulate interior lights
    .composite([{
      input: Buffer.from(
        `<svg width="100%" height="100%">
          <defs>
            <linearGradient id="dusk" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" style="stop-color:rgb(30,40,80);stop-opacity:0.3" />
              <stop offset="60%" style="stop-color:rgb(50,40,60);stop-opacity:0.1" />
              <stop offset="100%" style="stop-color:rgb(20,20,40);stop-opacity:0.2" />
            </linearGradient>
          </defs>
          <rect width="100%" height="100%" fill="url(#dusk)" />
        </svg>`
      ),
      blend: 'overlay',
    }])
    .jpeg({
      quality: 85,
      chromaSubsampling: '4:4:4',
    })
    .toBuffer()

  return processed
}

/**
 * Basic image enhancement for single images
 */
async function enhanceImage(buffer: Buffer): Promise<Buffer> {
  const enhanced = await sharp(buffer)
    // Auto-level/normalize
    .normalize()
    // Enhance colors and exposure per target spec
    .modulate({
      brightness: 1.02,  // Very slight lift
      saturation: 1.08,  // +8% saturation
    })
    // Subtle contrast
    .linear(1.05, -5)
    // Light sharpening
    .sharpen({
      sigma: 0.6,
      m1: 0.4,
      m2: 0.4,
    })
    // High-quality output
    .jpeg({
      quality: 85,
      chromaSubsampling: '4:4:4',
    })
    .toBuffer()

  return enhanced
}
