'use client'

import { useState, useCallback } from 'react'

// Backend URL - switch between local Python backend and Vercel API
// Local: http://localhost:8000 (supports RAW files)
// Vercel: '' (empty = use /api/process, no RAW support)
const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || ''

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'hdr' | 'twilight'>('hdr')
  const [useLocalBackend, setUseLocalBackend] = useState(false)

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
    setError(null)

    try {
      // Create FormData with all images
      const formData = new FormData()
      files.forEach((file) => {
        formData.append('images', file)
      })

      // Determine API URL
      // Local backend: http://localhost:8000/process (RAW support)
      // Vercel API: /api/process (no RAW support)
      const apiUrl = useLocalBackend
        ? `http://localhost:8000/process?mode=${mode}`
        : `/api/process?mode=${mode}`

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
    setError(null)
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2">AutoHDR Clone</h1>
        <p className="text-gray-400">Open-source AI real estate photo editing</p>
      </div>

      {/* Backend Toggle */}
      <div className="flex justify-center mb-4">
        <button
          onClick={() => setUseLocalBackend(!useLocalBackend)}
          className={`text-xs px-3 py-1 rounded-full transition ${
            useLocalBackend
              ? 'bg-green-600/20 text-green-400 border border-green-600'
              : 'bg-gray-800 text-gray-500 border border-gray-700'
          }`}
        >
          {useLocalBackend ? '🟢 Local Backend (RAW support)' : '☁️ Cloud API (JPG/PNG only)'}
        </button>
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
              ? 'Drop bracket images here (2-9 exposures)'
              : 'Drop a daytime exterior photo'}
          </p>
          <p className="text-gray-500">or click to browse</p>
          <p className="text-gray-600 text-xs mt-3">
            JPG, PNG, TIFF, RAW, PDF, PSD • Canon, Nikon, Sony, Fuji + all cameras
          </p>
        </label>
      </div>

      {/* Selected Files */}
      {files.length > 0 && (
        <div className="mt-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">
              {files.length} image{files.length > 1 ? 's' : ''} selected
            </h2>
            <button
              onClick={() => setFiles([])}
              className="text-red-400 hover:text-red-300"
            >
              Clear all
            </button>
          </div>

          <div className="grid grid-cols-4 gap-4 mb-6">
            {files.map((file, i) => (
              <div key={i} className="aspect-video bg-gray-800 rounded-lg overflow-hidden relative">
                {file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf') ? (
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
              </div>
            ))}
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

          {/* Preview */}
          <div className="rounded-xl overflow-hidden bg-gray-800 mb-4">
            <img
              src={resultUrl}
              alt="Processed result"
              className="w-full h-auto"
            />
          </div>

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
        <p>Open source • <a href="https://github.com/memjar/autohdr-clone" className="text-blue-400 hover:underline">GitHub</a></p>
      </footer>
    </main>
  )
}
