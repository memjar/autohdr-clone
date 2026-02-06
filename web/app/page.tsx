'use client'

import { useState, useCallback, useRef } from 'react'

// Version for cache-busting verification
const APP_VERSION = 'v2.0.0'

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
  const [useLocalBackend, setUseLocalBackend] = useState(true)
  const [backendUrl, setBackendUrl] = useState('http://192.168.1.147:8000')
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

  const SUPPORTED_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
    '.tiff', '.tif', '.heic', '.heif',
    '.raw', '.cr2', '.cr3', '.crw', '.nef', '.nrw', '.arw', '.srf', '.sr2',
    '.dng', '.orf', '.rw2', '.pef', '.ptx', '.raf', '.erf', '.mrw',
    '.3fr', '.fff', '.iiq', '.rwl', '.srw', '.x3f',
  ]

  const isValidFile = (file: File) => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase()
    return file.type.startsWith('image/') || SUPPORTED_EXTENSIONS.includes(ext)
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
      const originalFile = files[Math.floor(files.length / 2)]
      if (originalFile.type.startsWith('image/') && !isRawFile(originalFile.name)) {
        setOriginalUrl(URL.createObjectURL(originalFile))
      }

      const formData = new FormData()
      files.forEach((file) => {
        formData.append('images', file)
      })

      const params = new URLSearchParams({
        mode,
        brightness: brightness.toString(),
        contrast: contrast.toString(),
        vibrance: vibrance.toString(),
        white_balance: whiteBalance.toString(),
      })

      const apiUrl = useLocalBackend
        ? `${backendUrl}/process?${params}`
        : `/api/process?${params}`

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Processing failed: ${response.status}`)
      }

      const blob = await response.blob()
      const url = URL.createObjectURL(blob)

      setResultUrl(url)
      setResult('Processing complete!')
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
    a.download = `hdrit_${mode}_${Date.now()}.jpg`
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
    <main className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000" />
      </div>

      <div className="relative z-10 p-8 max-w-5xl mx-auto">
        {/* Navigation */}
        <nav className="flex justify-between items-center mb-12">
          <a href="/" className="flex items-center gap-3 group">
            <div className="w-10 h-10 bg-gradient-to-br from-cyan-400 to-blue-500 rounded-xl flex items-center justify-center shadow-lg shadow-cyan-500/25 group-hover:shadow-cyan-500/40 transition-all">
              <span className="text-xl">⚡</span>
            </div>
            <span className="font-bold text-2xl bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent">
              HDRit
            </span>
          </a>
          <div className="flex items-center gap-6">
            <a href="/about" className="text-gray-400 hover:text-white transition font-medium">
              About
            </a>
            <a href="/pricing" className="text-gray-400 hover:text-white transition font-medium">
              Pricing
            </a>
            <a href="/dashboard" className="text-gray-400 hover:text-white transition font-medium">
              Dashboard
            </a>
            <a
              href="/dashboard"
              className="px-5 py-2.5 bg-gradient-to-r from-cyan-500 to-blue-500 text-white font-semibold rounded-xl hover:from-cyan-400 hover:to-blue-400 transition-all shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40"
            >
              Sign In
            </a>
          </div>
        </nav>

        {/* Hero Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full text-cyan-400 text-sm font-medium mb-6">
            <span className="w-2 h-2 bg-cyan-400 rounded-full animate-pulse" />
            AI-Powered Processing
          </div>
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-white via-gray-100 to-gray-300 bg-clip-text text-transparent">
            HDRit
          </h1>
          <p className="text-xl text-gray-400 mb-2">
            Professional real estate photo editing in seconds
          </p>
          <p className="text-cyan-400 font-medium">
            Pro Processor v4.7.0 • Smart Scene Detection • Full RAW Support
          </p>
        </div>

        {/* Backend Status */}
        <div className="flex justify-center mb-8">
          <button
            onClick={() => setUseLocalBackend(!useLocalBackend)}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all ${
              useLocalBackend
                ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                : 'bg-gray-800/50 text-gray-400 border border-gray-700'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${useLocalBackend ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
            {useLocalBackend ? 'Pro Engine Connected' : 'Cloud Processing'}
          </button>
          {useLocalBackend && (
            <button
              onClick={() => setShowBackendSettings(!showBackendSettings)}
              className="ml-2 p-2 text-gray-400 hover:text-white transition"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
            </button>
          )}
        </div>

        {showBackendSettings && useLocalBackend && (
          <div className="flex justify-center mb-6">
            <div className="flex items-center gap-3 bg-gray-900/80 backdrop-blur rounded-xl px-4 py-3 border border-gray-800">
              <span className="text-sm text-gray-400">Engine URL:</span>
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                className="text-sm bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 w-56 text-white focus:border-cyan-500 outline-none transition"
              />
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${backendUrl}/health`)
                    const data = await res.json()
                    alert(`Connected! Version: ${data.components?.pro_processor?.version || 'Unknown'}`)
                  } catch (e) {
                    alert('Connection failed. Check URL and server status.')
                  }
                }}
                className="px-3 py-1.5 bg-cyan-500 text-white text-sm rounded-lg hover:bg-cyan-400 transition font-medium"
              >
                Test
              </button>
            </div>
          </div>
        )}

        {/* Mode Toggle */}
        <div className="flex justify-center gap-3 mb-10">
          <button
            onClick={() => setMode('hdr')}
            className={`px-8 py-3 rounded-xl font-semibold transition-all ${
              mode === 'hdr'
                ? 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/25'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 border border-gray-700'
            }`}
          >
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              HDR Merge
            </span>
          </button>
          <button
            onClick={() => setMode('twilight')}
            className={`px-8 py-3 rounded-xl font-semibold transition-all ${
              mode === 'twilight'
                ? 'bg-gradient-to-r from-orange-500 to-pink-500 text-white shadow-lg shadow-orange-500/25'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 border border-gray-700'
            }`}
          >
            <span className="flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
              Day to Dusk
            </span>
          </button>
        </div>

        {/* Upload Area */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="relative group"
        >
          <div className="absolute -inset-1 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-2xl blur opacity-20 group-hover:opacity-30 transition" />
          <div className="relative bg-gray-900/80 backdrop-blur border-2 border-dashed border-gray-700 group-hover:border-cyan-500/50 rounded-2xl p-12 text-center transition-all cursor-pointer">
            <input
              type="file"
              multiple={mode === 'hdr'}
              accept="image/*,.raw,.cr2,.cr3,.crw,.nef,.nrw,.arw,.srf,.sr2,.dng,.orf,.rw2,.pef,.ptx,.raf,.tiff,.tif,.heic,.heif"
              onChange={handleFileSelect}
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer block">
              <div className="w-20 h-20 mx-auto mb-6 bg-gradient-to-br from-cyan-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center border border-cyan-500/30">
                <svg className="w-10 h-10 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <p className="text-2xl font-semibold text-white mb-2">
                {mode === 'hdr' ? 'Drop your photos here' : 'Drop a daytime photo'}
              </p>
              <p className="text-gray-400 mb-4">or click to browse</p>
              {mode === 'hdr' && (
                <div className="inline-flex items-center gap-2 px-4 py-2 bg-cyan-500/10 border border-cyan-500/20 rounded-full text-cyan-400 text-sm">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Upload 3-15+ photos • AI auto-groups by scene
                </div>
              )}
              <p className="text-gray-500 text-sm mt-4">
                RAW, JPG, PNG, TIFF • Sony, Canon, Nikon, Fuji + all cameras
              </p>
            </label>
          </div>
        </div>

        {/* Selected Files */}
        {files.length > 0 && (
          <div className="mt-10">
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-white">
                  {files.length} photo{files.length > 1 ? 's' : ''} ready
                </h2>
                {files.length > 5 && mode === 'hdr' && (
                  <p className="text-cyan-400 text-sm mt-1 flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                    AI will detect ~{Math.ceil(files.length / 3)} scene{Math.ceil(files.length / 3) > 1 ? 's' : ''} automatically
                  </p>
                )}
              </div>
              <button
                onClick={() => setFiles([])}
                className="text-red-400 hover:text-red-300 text-sm font-medium"
              >
                Clear all
              </button>
            </div>

            <div className="grid grid-cols-5 gap-3 mb-8">
              {files.slice(0, 10).map((file, i) => (
                <div key={i} className="aspect-video bg-gray-800 rounded-xl overflow-hidden relative group border border-gray-700">
                  {isRawFile(file.name) ? (
                    <div className="w-full h-full flex flex-col items-center justify-center bg-gradient-to-br from-gray-800 to-gray-900">
                      <span className="text-2xl mb-1">📷</span>
                      <span className="text-[10px] text-cyan-400 font-medium">
                        {file.name.split('.').pop()?.toUpperCase()}
                      </span>
                    </div>
                  ) : file.type.startsWith('image/') ? (
                    <img
                      src={URL.createObjectURL(file)}
                      alt={file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      <span className="text-2xl">🖼️</span>
                    </div>
                  )}
                </div>
              ))}
              {files.length > 10 && (
                <div className="aspect-video bg-gray-800/50 rounded-xl flex items-center justify-center border border-gray-700">
                  <span className="text-gray-400 font-medium">+{files.length - 10} more</span>
                </div>
              )}
            </div>

            {/* Adjustment Sliders */}
            <div className="bg-gray-900/50 backdrop-blur rounded-xl p-6 mb-8 border border-gray-800">
              <h3 className="text-sm font-semibold text-gray-300 mb-4 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                </svg>
                Fine-tune
              </h3>
              <div className="grid grid-cols-2 gap-6">
                {[
                  { label: 'Brightness', value: brightness, setter: setBrightness },
                  { label: 'Contrast', value: contrast, setter: setContrast },
                  { label: 'Vibrance', value: vibrance, setter: setVibrance },
                  { label: 'White Balance', value: whiteBalance, setter: setWhiteBalance },
                ].map(({ label, value, setter }) => (
                  <div key={label}>
                    <div className="flex justify-between text-xs text-gray-400 mb-2">
                      <span>{label}</span>
                      <span className="text-cyan-400">{value > 0 ? '+' : ''}{value.toFixed(1)}</span>
                    </div>
                    <input
                      type="range"
                      min="-2"
                      max="2"
                      step="0.1"
                      value={value}
                      onChange={(e) => setter(parseFloat(e.target.value))}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
                    />
                  </div>
                ))}
              </div>
            </div>

            <button
              onClick={processImages}
              disabled={processing || (mode === 'hdr' && files.length < 2)}
              className="w-full py-4 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 disabled:from-gray-700 disabled:to-gray-700 disabled:cursor-not-allowed rounded-xl font-semibold text-lg transition-all shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/40 disabled:shadow-none"
            >
              {processing ? (
                <span className="flex items-center justify-center gap-3">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Processing with AI...
                </span>
              ) : mode === 'hdr' ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  {files.length > 5 ? `Process ${Math.ceil(files.length / 3)} Scenes` : `Merge ${files.length} Photos`}
                </span>
              ) : (
                'Convert to Twilight'
              )}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mt-8 p-6 bg-red-900/20 border border-red-500/30 rounded-xl">
            <p className="text-red-400 flex items-center gap-2">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              {error}
            </p>
          </div>
        )}

        {/* Result */}
        {resultUrl && (
          <div className="mt-10">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold text-emerald-400 flex items-center gap-2">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Processing Complete
              </h2>
              <button onClick={resetAll} className="text-gray-400 hover:text-white text-sm font-medium">
                Start Over
              </button>
            </div>

            <div className="rounded-2xl overflow-hidden bg-gray-900 border border-gray-800 mb-6">
              <img src={resultUrl} alt="Processed result" className="w-full h-auto" />
            </div>

            <button
              onClick={downloadResult}
              className="w-full py-4 bg-gradient-to-r from-emerald-500 to-green-500 hover:from-emerald-400 hover:to-green-400 rounded-xl font-semibold text-lg transition-all shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40 flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download Result
            </button>
          </div>
        )}

        {/* Features */}
        <div className="mt-20 grid grid-cols-3 gap-6">
          {[
            { icon: '⚡', title: 'HDR Merge', desc: 'Blend brackets instantly', gradient: 'from-cyan-500/20 to-blue-500/20' },
            { icon: '🌙', title: 'Day to Dusk', desc: 'Twilight conversion', gradient: 'from-orange-500/20 to-pink-500/20' },
            { icon: '🧠', title: 'Smart Scenes', desc: 'Auto-detect & group', gradient: 'from-purple-500/20 to-pink-500/20' },
          ].map(({ icon, title, desc, gradient }) => (
            <div key={title} className={`bg-gradient-to-br ${gradient} rounded-2xl p-6 border border-gray-800 text-center`}>
              <div className="text-4xl mb-3">{icon}</div>
              <h3 className="font-semibold text-white mb-1">{title}</h3>
              <p className="text-gray-400 text-sm">{desc}</p>
            </div>
          ))}
        </div>

        {/* Footer */}
        <footer className="mt-20 pt-8 border-t border-gray-800 text-center">
          <p className="text-gray-400">
            HDRit • Made by <a href="https://linky.my" className="text-cyan-400 hover:text-cyan-300 transition">Virul</a>
          </p>
          <p className="mt-2 text-xs text-gray-600">
            {APP_VERSION} • Pro Processor v4.7.0
          </p>
        </footer>
      </div>
    </main>
  )
}
