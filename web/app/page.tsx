'use client'

import { useState, useCallback, useRef } from 'react'

// Version for cache-busting verification
const APP_VERSION = 'v1.3.0'

// RAW file extensions (browsers can't display these)
const RAW_EXTENSIONS = [
  '.arw', '.srf', '.sr2',  // Sony
  '.cr2', '.cr3', '.crw',  // Canon
  '.nef', '.nrw',          // Nikon
  '.dng',                  // Adobe
  '.orf',                  // Olympus
  '.rw2',                  // Panasonic
  '.pef', '.ptx',          // Pentax
  '.raf',                  // Fujifilm
  '.raw', '.3fr', '.fff', '.iiq', '.rwl', '.srw', '.x3f',
]

// Check if file is a RAW format
const isRawFile = (filename: string): boolean => {
  const ext = '.' + filename.split('.').pop()?.toLowerCase()
  return RAW_EXTENSIONS.includes(ext)
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'hdr' | 'twilight'>('hdr')
  const [useLocalBackend, setUseLocalBackend] = useState(true) // Default to local for RAW support
  const [backendUrl, setBackendUrl] = useState('http://192.168.1.147:8000') // Mac Studio Pro Processor
  const [showBackendSettings, setShowBackendSettings] = useState(false)

  // Adjustment sliders state
  const [brightness, setBrightness] = useState(0)
  const [contrast, setContrast] = useState(0)
  const [vibrance, setVibrance] = useState(0)
  const [whiteBalance, setWhiteBalance] = useState(0)

  // Before/after comparison
  const [comparePosition, setComparePosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const compareRef = useRef<HTMLDivElement>(null)

  // All supported file extensions for photographers/realtors
  const SUPPORTED_EXTENSIONS = [
    // Standard images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
    // High quality
    '.tiff', '.tif', '.heic', '.heif',
    // RAW formats (all major camera brands)
    '.raw', '.cr2', '.cr3', '.crw',           // Canon
    '.nef', '.nrw',                            // Nikon
    '.arw', '.srf', '.sr2',                    // Sony
    '.dng',                                    // Adobe Universal RAW
    '.orf',                                    // Olympus
    '.rw2',                                    // Panasonic
    '.pef', '.ptx',                            // Pentax
    '.raf',                                    // Fujifilm
    '.erf',                                    // Epson
    '.mrw',                                    // Minolta
    '.3fr', '.fff',                            // Hasselblad
    '.iiq',                                    // Phase One
    '.rwl',                                    // Leica
    '.srw',                                    // Samsung
    '.x3f',                                    // Sigma
    // Documents
    '.pdf',
    // Photoshop/Design
    '.psd', '.psb', '.ai', '.eps',
  ]

  const isValidFile = (file: File) => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    return file.type.startsWith('image/') ||
           file.type === 'application/pdf' ||
           SUPPORTED_EXTENSIONS.includes(ext)
  }

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files).filter(isValidFile)
    setFiles(prev => [...prev, ...droppedFiles])
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const validFiles = Array.from(e.target.files).filter(isValidFile)
      setFiles(prev => [...prev, ...validFiles])
    }
  }

  const processImages = async () => {
    if (files.length === 0) return

    setProcessing(true)
    setResult(null)
    setResultUrl(null)
    setOriginalUrl(null)
    setError(null)

    try {
      // Store original for comparison (use first file or middle exposure)
      const originalFile = files[Math.floor(files.length / 2)]
      if (originalFile.type.startsWith('image/') && !isRawFile(originalFile.name)) {
        setOriginalUrl(URL.createObjectURL(originalFile))
      }

      // Create FormData with all images
      const formData = new FormData()
      files.forEach((file) => {
        formData.append('images', file)
      })

      // Build query params with adjustments
      const params = new URLSearchParams({
        mode,
        brightness: brightness.toString(),
        contrast: contrast.toString(),
        vibrance: vibrance.toString(),
        white_balance: whiteBalance.toString(),
      })

      // Determine API URL
      // Local backend: Mac Studio Pro Processor v3.1 (RAW support, 95%+ quality)
      // Vercel API: /api/process (basic Sharp processing, no RAW)
      const apiUrl = useLocalBackend
        ? `${backendUrl}/process?${params}`
        : `/api/process?${params}`

      // Call the processing API
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Processing failed: ${response.status}`)
      }

      // Get the processed image blob
      const blob = await response.blob()
      const url = URL.createObjectURL(blob)

      setResultUrl(url)
      setResult('Processing complete! Click to download.')
    } catch (err: any) {
      console.error('Processing error:', err)
      setError(err.message || 'Processing failed')
    } finally {
      setProcessing(false)
    }
  }

  const downloadResult = () => {
    if (!resultUrl) return
    const a = document.createElement('a')
    a.href = resultUrl
    a.download = `processed_${mode}_${Date.now()}.jpg`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const resetAll = () => {
    setFiles([])
    setResult(null)
    setResultUrl(null)
    setOriginalUrl(null)
    setError(null)
    setBrightness(0)
    setContrast(0)
    setVibrance(0)
    setWhiteBalance(0)
    setComparePosition(50)
  }

  // Handle comparison slider drag
  const handleCompareMouseDown = () => setIsDragging(true)
  const handleCompareMouseUp = () => setIsDragging(false)
  const handleCompareMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (!isDragging || !compareRef.current) return
    const rect = compareRef.current.getBoundingClientRect()
    const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
    const x = clientX - rect.left
    const percent = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setComparePosition(percent)
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      {/* Navigation */}
      <nav className="flex justify-between items-center mb-8">
        <div className="flex items-center gap-2">
          <span className="text-2xl">📸</span>
          <span className="font-bold text-xl">HDR it</span>
        </div>
        <div className="flex items-center gap-4">
          <a
            href="/about"
            className="text-gray-400 hover:text-cyan-400 transition"
          >
            About
          </a>
          <a
            href="/pricing"
            className="text-gray-400 hover:text-cyan-400 transition"
          >
            Pricing
          </a>
          <a
            href="/dashboard"
            className="text-gray-400 hover:text-cyan-400 transition"
          >
            Dashboard
          </a>
          <a
            href="/dashboard"
            className="px-4 py-2 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition"
          >
            Sign In
          </a>
        </div>
      </nav>

      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2">HDR it</h1>
        <p className="text-gray-400">AI-powered real estate photo editing • Professional quality</p>
        <p className="text-cyan-400 text-sm mt-2">Pro Processor v4.7.0 • Mertens Fusion • Full RAW Support</p>
      </div>

      {/* Backend Toggle */}
      <div className="flex flex-col items-center gap-2 mb-4">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setUseLocalBackend(!useLocalBackend)}
            className={`text-xs px-3 py-1 rounded-full transition ${
              useLocalBackend
                ? 'bg-green-600/20 text-green-400 border border-green-600'
                : 'bg-gray-800 text-gray-500 border border-gray-700'
            }`}
          >
            {useLocalBackend ? '🟢 Pro Processor v4.7.0 (RAW support)' : '☁️ Cloud API (JPG/PNG only)'}
          </button>
          {useLocalBackend && (
            <button
              onClick={() => setShowBackendSettings(!showBackendSettings)}
              className="text-xs px-2 py-1 text-gray-400 hover:text-cyan-400 transition"
            >
              ⚙️
            </button>
          )}
        </div>
        {useLocalBackend && showBackendSettings && (
          <div className="flex items-center gap-2 bg-gray-900 rounded-lg px-3 py-2">
            <span className="text-xs text-gray-400">Backend URL:</span>
            <input
              type="text"
              value={backendUrl}
              onChange={(e) => setBackendUrl(e.target.value)}
              placeholder="http://192.168.1.147:8000"
              className="text-xs bg-gray-800 border border-gray-700 rounded px-2 py-1 w-48 text-white focus:border-cyan-400 outline-none"
            />
            <button
              onClick={async () => {
                try {
                  const res = await fetch(`${backendUrl}/health`)
                  const data = await res.json()
                  alert(`✓ Connected!\nProcessor: Pro v${data.components?.pro_processor?.version || '?'}\nQuality: ${data.quality_level}`)
                } catch (e) {
                  alert('✗ Cannot connect to backend. Check the URL and ensure the server is running.')
                }
              }}
              className="text-xs px-2 py-1 bg-cyan-400 text-black rounded hover:bg-cyan-300 transition"
            >
              Test
            </button>
          </div>
        )}
      </div>

      {/* Mode Toggle */}
      <div className="flex justify-center gap-4 mb-8">
        <button
          onClick={() => setMode('hdr')}
          className={`px-6 py-2 rounded-lg transition ${
            mode === 'hdr'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          HDR Merge
        </button>
        <button
          onClick={() => setMode('twilight')}
          className={`px-6 py-2 rounded-lg transition ${
            mode === 'twilight'
              ? 'bg-orange-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Day to Dusk
        </button>
      </div>

      {/* Upload Area */}
      <div
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
        className="border-2 border-dashed border-gray-700 rounded-xl p-12 text-center hover:border-gray-500 transition cursor-pointer"
      >
        <input
          type="file"
          multiple={mode === 'hdr'}
          accept="image/*,application/pdf,.raw,.cr2,.cr3,.crw,.nef,.nrw,.arw,.srf,.sr2,.dng,.orf,.rw2,.pef,.ptx,.raf,.erf,.mrw,.3fr,.fff,.iiq,.rwl,.srw,.x3f,.tiff,.tif,.heic,.heif,.psd,.psb,.ai,.eps"
          onChange={handleFileSelect}
          className="hidden"
          id="file-input"
        />
        <label htmlFor="file-input" className="cursor-pointer">
          <div className="text-6xl mb-4">
            {mode === 'hdr' ? '📷' : '🌅'}
          </div>
          <p className="text-xl mb-2">
            {mode === 'hdr'
              ? 'Drop your bracket photos here'
              : 'Drop a daytime exterior photo'}
          </p>
          <p className="text-gray-500">or click to browse</p>
          {mode === 'hdr' && (
            <p className="text-cyan-400/80 text-sm mt-2">
              Upload 3-15+ photos • Auto-groups by scene • Multiple rooms supported
            </p>
          )}
          <p className="text-gray-600 text-xs mt-3">
            JPG, PNG, TIFF, RAW • Canon, Nikon, Sony, Fuji + all cameras
          </p>
        </label>
      </div>

      {/* Selected Files */}
      {files.length > 0 && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-lg font-semibold">
                {files.length} image{files.length > 1 ? 's' : ''} selected
              </h2>
              {files.length > 5 && mode === 'hdr' && (
                <p className="text-cyan-400 text-xs mt-1">
                  🧠 AI will auto-detect ~{Math.ceil(files.length / 3)} scene{Math.ceil(files.length / 3) > 1 ? 's' : ''} and group brackets
                </p>
              )}
            </div>
            <button
              onClick={() => setFiles([])}
              className="text-red-400 hover:text-red-300"
            >
              Clear all
            </button>
          </div>

          <div className="grid grid-cols-4 gap-4 mb-6">
            {files.map((file, i) => (
              <div key={i} className="aspect-video bg-gray-800 rounded-lg overflow-hidden relative group">
                {/* RAW files - show camera icon */}
                {isRawFile(file.name) ? (
                  <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-700 to-gray-800">
                    <div className="text-4xl mb-2">📷</div>
                    <span className="text-xs text-gray-300 font-medium px-2 py-1 bg-gray-900/50 rounded">
                      {file.name.split('.').pop()?.toUpperCase()}
                    </span>
                    <span className="text-[10px] text-gray-500 mt-1 truncate max-w-full px-2">
                      {file.name}
                    </span>
                  </div>
                ) : file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf') ? (
                  <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
                    <span className="text-3xl">📄</span>
                    <span className="text-xs mt-1 truncate max-w-full px-2">{file.name}</span>
                  </div>
                ) : file.type.startsWith('image/') ? (
                  <img
                    src={URL.createObjectURL(file)}
                    alt={file.name}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
                    <span className="text-3xl">🖼️</span>
                    <span className="text-xs mt-1 truncate max-w-full px-2">{file.name}</span>
                  </div>
                )}
                {/* Exposure indicator for brackets */}
                <div className="absolute bottom-1 right-1 text-[10px] text-gray-400 bg-black/50 px-1 rounded">
                  {i === 0 ? '−EV' : i === files.length - 1 ? '+EV' : '0'}
                </div>
              </div>
            ))}
          </div>

          {/* Adjustment Sliders */}
          <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-4">Adjustments</h3>
            <div className="grid grid-cols-2 gap-4">
              {/* Brightness */}
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Brightness</span>
                  <span>{brightness > 0 ? '+' : ''}{brightness.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={brightness}
                  onChange={(e) => setBrightness(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
              {/* Contrast */}
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Contrast</span>
                  <span>{contrast > 0 ? '+' : ''}{contrast.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={contrast}
                  onChange={(e) => setContrast(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
              {/* Vibrance */}
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>Vibrance</span>
                  <span>{vibrance > 0 ? '+' : ''}{vibrance.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={vibrance}
                  onChange={(e) => setVibrance(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
              {/* White Balance */}
              <div>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
                  <span>White Balance</span>
                  <span>{whiteBalance > 0 ? '+' : ''}{whiteBalance.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={whiteBalance}
                  onChange={(e) => setWhiteBalance(parseFloat(e.target.value))}
                  className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider-thumb"
                />
              </div>
            </div>
            {/* Reset button */}
            {(brightness !== 0 || contrast !== 0 || vibrance !== 0 || whiteBalance !== 0) && (
              <button
                onClick={() => { setBrightness(0); setContrast(0); setVibrance(0); setWhiteBalance(0); }}
                className="mt-3 text-xs text-gray-500 hover:text-gray-300 transition"
              >
                Reset adjustments
              </button>
            )}
          </div>

          <button
            onClick={processImages}
            disabled={processing || (mode === 'hdr' && files.length < 2)}
            className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-semibold transition"
          >
            {processing ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Processing...
              </span>
            ) : mode === 'hdr' ? (
              `Merge ${files.length} Images`
            ) : (
              'Convert to Twilight'
            )}
          </button>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-8 p-6 bg-red-900/30 border border-red-700 rounded-lg">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Result */}
      {resultUrl && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold text-green-400">Processing Complete!</h2>
            <button
              onClick={resetAll}
              className="text-gray-400 hover:text-gray-300"
            >
              Start Over
            </button>
          </div>

          {/* Before/After Comparison */}
          {originalUrl ? (
            <div
              ref={compareRef}
              className="relative rounded-xl overflow-hidden bg-gray-800 mb-4 cursor-ew-resize select-none"
              onMouseMove={handleCompareMove}
              onMouseUp={handleCompareMouseUp}
              onMouseLeave={handleCompareMouseUp}
              onTouchMove={handleCompareMove}
              onTouchEnd={handleCompareMouseUp}
            >
              {/* After (processed) - full width */}
              <img
                src={resultUrl}
                alt="Processed result"
                className="w-full h-auto"
                draggable={false}
              />
              {/* Before (original) - clipped */}
              <div
                className="absolute inset-0 overflow-hidden"
                style={{ width: `${comparePosition}%` }}
              >
                <img
                  src={originalUrl}
                  alt="Original"
                  className="w-full h-auto"
                  style={{ width: `${100 / (comparePosition / 100)}%`, maxWidth: 'none' }}
                  draggable={false}
                />
              </div>
              {/* Slider handle */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-white cursor-ew-resize"
                style={{ left: `${comparePosition}%`, transform: 'translateX(-50%)' }}
                onMouseDown={handleCompareMouseDown}
                onTouchStart={handleCompareMouseDown}
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-8 h-8 bg-white rounded-full shadow-lg flex items-center justify-center">
                  <svg className="w-4 h-4 text-gray-800" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 9l4-4 4 4m0 6l-4 4-4-4" />
                  </svg>
                </div>
              </div>
              {/* Labels */}
              <div className="absolute top-2 left-2 px-2 py-1 bg-black/60 rounded text-xs text-white">
                Before
              </div>
              <div className="absolute top-2 right-2 px-2 py-1 bg-black/60 rounded text-xs text-white">
                After
              </div>
            </div>
          ) : (
            /* Simple preview when no original available */
            <div className="rounded-xl overflow-hidden bg-gray-800 mb-4">
              <img
                src={resultUrl}
                alt="Processed result"
                className="w-full h-auto"
              />
            </div>
          )}

          {/* Download Button */}
          <button
            onClick={downloadResult}
            className="w-full py-3 bg-green-600 hover:bg-green-500 rounded-lg font-semibold transition flex items-center justify-center gap-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            Download Processed Image
          </button>
        </div>
      )}

      {/* Features */}
      <div className="mt-16 grid grid-cols-3 gap-8 text-center">
        <div>
          <div className="text-3xl mb-2">🔀</div>
          <h3 className="font-semibold mb-1">HDR Merge</h3>
          <p className="text-gray-500 text-sm">Blend bracketed exposures</p>
        </div>
        <div>
          <div className="text-3xl mb-2">🌙</div>
          <h3 className="font-semibold mb-1">Day to Dusk</h3>
          <p className="text-gray-500 text-sm">Twilight conversion</p>
        </div>
        <div>
          <div className="text-3xl mb-2">✨</div>
          <h3 className="font-semibold mb-1">Sky Replace</h3>
          <p className="text-gray-500 text-sm">Coming soon</p>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-16 pt-8 border-t border-gray-800 text-center text-gray-500 text-sm">
        <p>HDR it • <a href="https://github.com/memjar/autohdr-clone" className="text-blue-400 hover:underline">GitHub</a></p>
        <p className="mt-2 text-xs text-gray-600">
          {APP_VERSION} • Backend: {useLocalBackend ? '🟢 Pro Processor v4.7.0' : '☁️ Cloud API'}
        </p>
      </footer>
    </main>
  )
}
