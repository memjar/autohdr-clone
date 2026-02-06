import { NextRequest, NextResponse } from 'next/server'
import sharp from 'sharp'

export const runtime = 'nodejs'
export const maxDuration = 60

/**
 * AutoHDR Clone - Image Processing API
 *
 * Implements HDR-like processing pipeline:
 * 1. HDR Tone Mapping (shadow boost + highlight compression)
 * 2. Color adjustments (brightness, contrast, vibrance, white balance)
 * 3. Sharpening and clarity
 *
 * POST /api/process
 * Body: FormData with 'images' field
 * Query: ?mode=hdr|twilight&brightness=0&contrast=0&vibrance=0&whiteBalance=0
 */
export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData()
    const searchParams = request.nextUrl.searchParams

    // Parse settings from query params
    const mode = searchParams.get('mode') || 'hdr'
    const brightness = parseFloat(searchParams.get('brightness') || '0')
    const contrast = parseFloat(searchParams.get('contrast') || '0')
    const vibrance = parseFloat(searchParams.get('vibrance') || '0')
    const whiteBalance = parseFloat(searchParams.get('whiteBalance') || '0')

    // Get uploaded images
    const imageFiles: File[] = []
    const entries = Array.from(formData.entries())
    for (const [key, value] of entries) {
      if (key === 'images' && value instanceof File) {
        imageFiles.push(value)
      }
    }

    if (imageFiles.length === 0) {
      return NextResponse.json({ error: 'No images provided' }, { status: 400 })
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
      processedBuffer = await processTwilight(buffers[0], { brightness, contrast, vibrance, whiteBalance })
    } else {
      processedBuffer = await processHDR(buffers, { brightness, contrast, vibrance, whiteBalance })
    }

    return new NextResponse(new Uint8Array(processedBuffer), {
      headers: {
        'Content-Type': 'image/jpeg',
        'Content-Disposition': `attachment; filename="autohdr_${Date.now()}.jpg"`,
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

interface Settings {
  brightness: number  // -2 to +2
  contrast: number    // -2 to +2
  vibrance: number    // -2 to +2
  whiteBalance: number // -2 to +2
}

/**
 * HDR Processing Pipeline - BRACKET MERGING
 *
 * Takes 2-9 bracketed exposures and merges them:
 * - Dark exposure: preserves highlights (windows, sky)
 * - Normal exposure: balanced mid-tones
 * - Bright exposure: recovers shadows (dark corners, under furniture)
 *
 * Uses luminosity-based blending (Mertens-style exposure fusion)
 */
async function processHDR(buffers: Buffer[], settings: Settings): Promise<Buffer> {
  // Single image? Just enhance it
  if (buffers.length === 1) {
    return processSingleHDR(buffers[0], settings)
  }

  // ==========================================
  // STAGE 0: ANALYZE & SORT BRACKETS BY BRIGHTNESS
  // ==========================================
  const analyzed = await Promise.all(
    buffers.map(async (buf, index) => {
      const { data, info } = await sharp(buf)
        .resize(100, 100, { fit: 'inside' }) // Small sample for speed
        .raw()
        .toBuffer({ resolveWithObject: true })

      // Calculate average brightness
      let sum = 0
      for (let i = 0; i < data.length; i += 3) {
        sum += (data[i] + data[i + 1] + data[i + 2]) / 3
      }
      const avgBrightness = sum / (data.length / 3)

      return { buffer: buf, brightness: avgBrightness, index }
    })
  )

  // Sort by brightness: dark → normal → bright
  analyzed.sort((a, b) => a.brightness - b.brightness)

  const darkBuffer = analyzed[0].buffer                           // Darkest - has highlight detail
  const brightBuffer = analyzed[analyzed.length - 1].buffer       // Brightest - has shadow detail
  const normalBuffer = analyzed[Math.floor(analyzed.length / 2)].buffer // Middle

  // ==========================================
  // STAGE 1: EXTRACT RAW PIXELS FROM ALL BRACKETS
  // ==========================================
  const [darkImg, normalImg, brightImg] = await Promise.all([
    sharp(darkBuffer).raw().toBuffer({ resolveWithObject: true }),
    sharp(normalBuffer).raw().toBuffer({ resolveWithObject: true }),
    sharp(brightBuffer).raw().toBuffer({ resolveWithObject: true }),
  ])

  const { width, height, channels } = normalImg.info
  const darkData = darkImg.data
  const normalData = normalImg.data
  const brightData = brightImg.data

  // ==========================================
  // STAGE 2: LUMINOSITY-BASED EXPOSURE FUSION
  // ==========================================
  // For each pixel:
  // - If dark (shadow): use bright exposure
  // - If bright (highlight): use dark exposure
  // - Mid-tones: blend all three

  const merged = Buffer.alloc(normalData.length)

  for (let i = 0; i < normalData.length; i += channels) {
    // Calculate luminance of normal exposure pixel
    const r = normalData[i]
    const g = normalData[i + 1]
    const b = normalData[i + 2]
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255

    // Weight calculation (Mertens-style)
    // Shadows (luminance < 0.3): favor bright exposure
    // Highlights (luminance > 0.7): favor dark exposure
    // Mid-tones: blend all

    let darkWeight: number, normalWeight: number, brightWeight: number

    if (luminance < 0.25) {
      // Shadow region - use bright exposure for detail
      darkWeight = 0.1
      normalWeight = 0.2
      brightWeight = 0.7
    } else if (luminance > 0.75) {
      // Highlight region - use dark exposure to prevent blowout
      darkWeight = 0.7
      normalWeight = 0.2
      brightWeight = 0.1
    } else {
      // Mid-tone - blend based on distance from center
      const distFromMid = Math.abs(luminance - 0.5) * 2
      normalWeight = 0.5 + (1 - distFromMid) * 0.3
      darkWeight = (1 - normalWeight) * (luminance > 0.5 ? 0.7 : 0.3)
      brightWeight = (1 - normalWeight) - darkWeight
    }

    // Normalize weights
    const totalWeight = darkWeight + normalWeight + brightWeight

    // Blend each channel
    for (let c = 0; c < Math.min(channels, 3); c++) {
      const blended = (
        darkData[i + c] * (darkWeight / totalWeight) +
        normalData[i + c] * (normalWeight / totalWeight) +
        brightData[i + c] * (brightWeight / totalWeight)
      )
      merged[i + c] = Math.round(Math.min(255, Math.max(0, blended)))
    }

    // Alpha channel if present
    if (channels === 4) {
      merged[i + 3] = normalData[i + 3]
    }
  }

  // ==========================================
  // STAGE 3: RECONSTRUCT IMAGE FROM MERGED DATA
  // ==========================================
  let processed = sharp(merged, {
    raw: {
      width,
      height,
      channels: channels as 3 | 4,
    }
  })

  // Apply HDR tone mapping on merged result
  processed = processed.normalize()

  // ==========================================
  // STAGE 2: ADJUSTMENTS
  // ==========================================

  // Calculate modulation values
  // Brightness: -2 to +2 maps to 0.8 to 1.2
  const brightnessFactor = 1 + (settings.brightness * 0.1)

  // Vibrance: -2 to +2 maps to 0.8 to 1.2 saturation
  const saturationFactor = 1 + (settings.vibrance * 0.1)

  processed = processed.modulate({
    brightness: brightnessFactor,
    saturation: saturationFactor,
  })

  // Contrast: apply via linear transformation
  // -2 to +2 maps to multiplier 0.8 to 1.2
  if (settings.contrast !== 0) {
    const contrastFactor = 1 + (settings.contrast * 0.1)
    const offset = 128 * (1 - contrastFactor)
    processed = processed.linear(contrastFactor, offset)
  }

  // ==========================================
  // STAGE 3: WHITE BALANCE (Color Temperature)
  // ==========================================
  if (settings.whiteBalance !== 0) {
    // Positive = warmer (more red/yellow), Negative = cooler (more blue)
    // Sharp uses tint which takes RGB values
    if (settings.whiteBalance > 0) {
      // Warmer - add red/orange tint
      const warmth = Math.round(settings.whiteBalance * 15)
      processed = processed.tint({
        r: 255,
        g: 240 - warmth,
        b: 220 - warmth * 2
      })
    } else {
      // Cooler - add blue tint
      const coolness = Math.round(Math.abs(settings.whiteBalance) * 15)
      processed = processed.tint({
        r: 220 - coolness * 2,
        g: 235 - coolness,
        b: 255
      })
    }
  }

  // ==========================================
  // STAGE 4: LOCAL CONTRAST (Clarity)
  // ==========================================
  // Unsharp mask with large radius = clarity/local contrast
  processed = processed.sharpen({
    sigma: 1.5,      // Radius for clarity effect
    m1: 0.8,         // Flat area sharpening
    m2: 0.4,         // Edge sharpening
  })

  // ==========================================
  // STAGE 5: FINAL SHARPENING
  // ==========================================
  processed = processed.sharpen({
    sigma: 0.5,      // Small radius for detail sharpening
    m1: 0.5,
    m2: 0.5,
  })

  // ==========================================
  // OUTPUT
  // ==========================================
  const result = await processed
    .jpeg({
      quality: 90,
      chromaSubsampling: '4:4:4',  // Best color quality
    })
    .toBuffer()

  return result
}

/**
 * Twilight (Day-to-Dusk) Processing
 *
 * Color temperature shift + darkening + warm glow simulation
 */
async function processTwilight(buffer: Buffer, settings: Settings): Promise<Buffer> {
  // Start with HDR-like base processing
  let processed = sharp(buffer)
    .normalize()

  // Apply standard adjustments
  const brightnessFactor = 1 + (settings.brightness * 0.1)
  const saturationFactor = 1 + (settings.vibrance * 0.1)

  processed = processed.modulate({
    brightness: brightnessFactor * 0.85,  // Darken for dusk
    saturation: saturationFactor * 1.15,  // Boost colors for sunset
  })

  // Warm/pink twilight tint
  processed = processed.tint({
    r: 255,
    g: 200,
    b: 170
  })

  // Slight contrast boost
  processed = processed.linear(1.05, -5)

  // Add warm overlay gradient (simulates sky glow)
  // Sharp doesn't do gradients easily, so we approximate with tint

  // Sharpening
  processed = processed.sharpen({
    sigma: 0.8,
    m1: 0.5,
    m2: 0.4,
  })

  const result = await processed
    .jpeg({
      quality: 90,
      chromaSubsampling: '4:4:4',
    })
    .toBuffer()

  return result
}

/**
 * Single Image HDR Enhancement
 *
 * When only 1 image is provided, apply HDR-style tone mapping:
 * - Shadow recovery via gamma
 * - Highlight compression
 * - Local contrast enhancement
 */
async function processSingleHDR(buffer: Buffer, settings: Settings): Promise<Buffer> {
  // ==========================================
  // STAGE 1: HDR TONE MAPPING (simulated)
  // ==========================================
  let processed = sharp(buffer)
    // Normalize - auto white/black point (recovers dynamic range)
    .normalize()
    // Gamma < 1 lifts shadows
    .gamma(0.9, 1.1)

  // ==========================================
  // STAGE 2: ADJUSTMENTS
  // ==========================================
  const brightnessFactor = 1 + (settings.brightness * 0.1)
  const saturationFactor = 1 + (settings.vibrance * 0.1)

  processed = processed.modulate({
    brightness: brightnessFactor,
    saturation: saturationFactor * 1.05, // Slight boost for HDR look
  })

  // Contrast
  if (settings.contrast !== 0) {
    const contrastFactor = 1 + (settings.contrast * 0.1)
    const offset = 128 * (1 - contrastFactor)
    processed = processed.linear(contrastFactor, offset)
  }

  // ==========================================
  // STAGE 3: WHITE BALANCE
  // ==========================================
  if (settings.whiteBalance !== 0) {
    if (settings.whiteBalance > 0) {
      const warmth = Math.round(settings.whiteBalance * 15)
      processed = processed.tint({ r: 255, g: 240 - warmth, b: 220 - warmth * 2 })
    } else {
      const coolness = Math.round(Math.abs(settings.whiteBalance) * 15)
      processed = processed.tint({ r: 220 - coolness * 2, g: 235 - coolness, b: 255 })
    }
  }

  // ==========================================
  // STAGE 4: LOCAL CONTRAST (Clarity)
  // ==========================================
  processed = processed.sharpen({ sigma: 1.5, m1: 0.8, m2: 0.4 })

  // ==========================================
  // STAGE 5: FINAL SHARPENING
  // ==========================================
  processed = processed.sharpen({ sigma: 0.5, m1: 0.5, m2: 0.5 })

  // ==========================================
  // OUTPUT
  // ==========================================
  return processed
    .jpeg({ quality: 90, chromaSubsampling: '4:4:4' })
    .toBuffer()
}
