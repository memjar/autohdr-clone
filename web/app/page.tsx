'use client'

import { useState, useCallback, useRef, useEffect } from 'react'

const APP_VERSION = 'v3.0.0'
const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'https://hdr.it.com.ngrok.pro'

const RAW_EXTENSIONS = [
  '.arw', '.srf', '.sr2', '.cr2', '.cr3', '.crw', '.nef', '.nrw',
  '.dng', '.orf', '.rw2', '.pef', '.ptx', '.raf',
  '.raw', '.3fr', '.fff', '.iiq', '.rwl', '.srw', '.x3f',
]

const isRawFile = (filename: string): boolean => {
  const ext = '.' + filename.split('.').pop()?.toLowerCase()
  return RAW_EXTENSIONS.includes(ext)
}

// SVG Icons as components
const Icons = {
  layers: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
  ),
  moon: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
    </svg>
  ),
  sparkles: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
    </svg>
  ),
  upload: (
    <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
  ),
  check: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
    </svg>
  ),
  photo: (
    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
    </svg>
  ),
  download: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
    </svg>
  ),
  settings: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
    </svg>
  ),
  zap: (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
    </svg>
  ),
}

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'hdr' | 'twilight' | 'enhance'>('hdr')
  const [useLocalBackend, setUseLocalBackend] = useState(true)
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL)
  const [showBackendSettings, setShowBackendSettings] = useState(false)
  const [proProcessorStatus, setProProcessorStatus] = useState<'checking' | 'connected' | 'unavailable'>('checking')
  const [proProcessorVersion, setProProcessorVersion] = useState<string | null>(null)

  // Adjustment sliders
  const [brightness, setBrightness] = useState(0)
  const [contrast, setContrast] = useState(0)
  const [vibrance, setVibrance] = useState(0)
  const [whiteBalance, setWhiteBalance] = useState(0)

  // Perfect Edit options
  const [windowPull, setWindowPull] = useState(true)
  const [skyEnhance, setSkyEnhance] = useState(true)
  const [perspectiveCorrect, setPerspectiveCorrect] = useState(true)
  const [noiseReduction, setNoiseReduction] = useState(true)
  const [sharpening, setSharpening] = useState(true)

  const [progress, setProgress] = useState(0)
  const [progressStatus, setProgressStatus] = useState('')
  const progressInterval = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    return () => {
      if (progressInterval.current) clearInterval(progressInterval.current)
    }
  }, [])

  useEffect(() => {
    const checkProProcessor = async () => {
      try {
        const controller = new AbortController()
        const timeout = setTimeout(() => controller.abort(), 3000)
        const res = await fetch(`${backendUrl}/health`, { signal: controller.signal, mode: 'cors' })
        clearTimeout(timeout)
        if (res.ok) {
          const data = await res.json()
          if (data.pro_processor_available) {
            setProProcessorStatus('connected')
            setProProcessorVersion(data.components?.pro_processor?.version || 'unknown')
            setUseLocalBackend(true)
          } else {
            setProProcessorStatus('unavailable')
            setUseLocalBackend(false)
          }
        } else {
          setProProcessorStatus('unavailable')
          setUseLocalBackend(false)
        }
      } catch {
        setProProcessorStatus('unavailable')
        setUseLocalBackend(false)
      }
    }
    checkProProcessor()
  }, [backendUrl])

  const SUPPORTED_EXTENSIONS = [
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif',
    ...RAW_EXTENSIONS
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

    if (!useLocalBackend) {
      const rawFiles = files.filter(f => isRawFile(f.name))
      if (rawFiles.length > 0) {
        setError(`RAW files require the Pro Engine. Please use JPG/PNG for cloud processing.`)
        return
      }
    }

    setProcessing(true)
    setResult(null)
    setResultUrl(null)
    setOriginalUrl(null)
    setError(null)
    setProgress(0)
    setProgressStatus('Uploading...')

    const stages = mode === 'enhance' ? [
      { pct: 15, msg: 'Analyzing image...' },
      { pct: 30, msg: 'Correcting exposure...' },
      { pct: 45, msg: 'Balancing colors...' },
      { pct: 60, msg: 'Enhancing details...' },
      { pct: 75, msg: 'Applying corrections...' },
      { pct: 90, msg: 'Finalizing...' },
    ] : mode === 'twilight' ? [
      { pct: 20, msg: 'Analyzing scene...' },
      { pct: 45, msg: 'Generating sky...' },
      { pct: 70, msg: 'Blending lights...' },
      { pct: 90, msg: 'Finalizing...' },
    ] : [
      { pct: 15, msg: 'Reading files...' },
      { pct: 35, msg: 'Aligning brackets...' },
      { pct: 55, msg: 'Merging exposures...' },
      { pct: 75, msg: 'Tone mapping...' },
      { pct: 90, msg: 'Finalizing...' },
    ]

    let stageIndex = 0
    progressInterval.current = setInterval(() => {
      if (stageIndex < stages.length) {
        setProgress(stages[stageIndex].pct)
        setProgressStatus(stages[stageIndex].msg)
        stageIndex++
      }
    }, 1500)

    try {
      const originalFile = files[Math.floor(files.length / 2)]
      if (originalFile.type.startsWith('image/') && !isRawFile(originalFile.name)) {
        setOriginalUrl(URL.createObjectURL(originalFile))
      }

      const formData = new FormData()
      files.forEach(file => formData.append('images', file))

      const params = new URLSearchParams({
        mode,
        brightness: brightness.toString(),
        contrast: contrast.toString(),
        vibrance: vibrance.toString(),
        white_balance: whiteBalance.toString(),
      })

      if (mode === 'enhance') {
        params.set('window_pull', windowPull.toString())
        params.set('sky_enhance', skyEnhance.toString())
        params.set('perspective_correct', perspectiveCorrect.toString())
        params.set('noise_reduction', noiseReduction.toString())
        params.set('sharpening', sharpening.toString())
      }

      const apiUrl = useLocalBackend ? `${backendUrl}/process?${params}` : `/api/process?${params}`
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000)

      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.error || `Processing failed: ${response.status}`)
      }

      const blob = await response.blob()
      if (blob.size < 1000) throw new Error('Invalid response from server')

      const url = URL.createObjectURL(blob)
      setProgress(100)
      setProgressStatus('Complete')
      setResultUrl(url)
      setResult('Processing complete!')
    } catch (err: any) {
      if (err.name === 'AbortError') {
        setError('Request timed out. Check if the backend is accessible.')
      } else if (err.message.includes('Failed to fetch')) {
        setError('Cannot connect to backend. Check your connection.')
      } else {
        setError(err.message || 'Processing failed')
      }
    } finally {
      if (progressInterval.current) {
        clearInterval(progressInterval.current)
        progressInterval.current = null
      }
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
    setProgress(0)
    setProgressStatus('')
  }

  return (
    <main className="min-h-screen bg-[#0a0a0f]">
      {/* Navigation */}
      <nav className="border-b border-white/5">
        <div className="max-w-6xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-16">
            <a href="/" className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <span className="text-white font-bold text-sm">H</span>
              </div>
              <span className="font-semibold text-white text-lg">HDRit</span>
            </a>
            <div className="flex items-center gap-6">
              <a href="/pricing" className="hidden sm:block text-sm text-gray-400 hover:text-white transition">Pricing</a>
              <a href="/dashboard" className="hidden sm:block text-sm text-gray-400 hover:text-white transition">Dashboard</a>
              <a href="/dashboard" className="px-4 py-2 text-sm font-medium text-white bg-white/10 hover:bg-white/15 rounded-lg transition">
                Sign In
              </a>
            </div>
          </div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8 sm:py-16">
        {/* Hero */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-blue-500/10 border border-blue-500/20 mb-6">
            <div className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
            <span className="text-xs font-medium text-blue-400">AI-Powered Photo Editing</span>
          </div>
          <h1 className="text-3xl sm:text-5xl lg:text-6xl font-bold text-white mb-4 tracking-tight">
            Professional Real Estate
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
              Photo Editing
            </span>
          </h1>
          <p className="text-gray-400 text-base sm:text-lg max-w-2xl mx-auto mb-8">
            Transform your property photos with AI-powered HDR blending,
            day-to-dusk conversion, and professional enhancements in seconds.
          </p>

          {/* Status Badge */}
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/10 text-sm">
            <div className={`w-2 h-2 rounded-full ${
              proProcessorStatus === 'connected' ? 'bg-green-400' :
              proProcessorStatus === 'checking' ? 'bg-yellow-400 animate-pulse' : 'bg-gray-500'
            }`} />
            <span className="text-gray-300">
              {proProcessorStatus === 'connected' ? `Pro Engine ${proProcessorVersion}` :
               proProcessorStatus === 'checking' ? 'Connecting...' : 'Cloud Processing'}
            </span>
            {proProcessorStatus === 'connected' && (
              <button onClick={() => setShowBackendSettings(!showBackendSettings)} className="text-gray-500 hover:text-gray-300">
                {Icons.settings}
              </button>
            )}
          </div>
        </div>

        {showBackendSettings && (
          <div className="max-w-md mx-auto mb-8 p-4 rounded-xl bg-white/5 border border-white/10">
            <div className="flex items-center gap-3">
              <input
                type="text"
                value={backendUrl}
                onChange={(e) => setBackendUrl(e.target.value)}
                className="flex-1 px-3 py-2 text-sm bg-white/5 border border-white/10 rounded-lg text-white focus:border-blue-500 focus:outline-none"
              />
              <button
                onClick={async () => {
                  try {
                    const res = await fetch(`${backendUrl}/health`)
                    const data = await res.json()
                    alert(`Connected! Version: ${data.components?.pro_processor?.version || 'Unknown'}`)
                  } catch {
                    alert('Connection failed')
                  }
                }}
                className="px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-500 rounded-lg transition"
              >
                Test
              </button>
            </div>
          </div>
        )}

        {/* Mode Selection */}
        <div className="flex flex-wrap justify-center gap-2 mb-8">
          {[
            { id: 'hdr', label: 'HDR Merge', icon: Icons.layers, color: 'blue' },
            { id: 'twilight', label: 'Day to Dusk', icon: Icons.moon, color: 'orange' },
            { id: 'enhance', label: 'Enhance', icon: Icons.sparkles, color: 'emerald' },
          ].map(({ id, label, icon, color }) => (
            <button
              key={id}
              onClick={() => setMode(id as typeof mode)}
              className={`flex items-center gap-2 px-4 sm:px-6 py-2.5 sm:py-3 rounded-xl text-sm font-medium transition-all ${
                mode === id
                  ? `bg-${color}-500/20 text-${color}-400 border border-${color}-500/30`
                  : 'bg-white/5 text-gray-400 border border-white/10 hover:bg-white/10 hover:text-white'
              }`}
              style={mode === id ? {
                backgroundColor: color === 'blue' ? 'rgba(59,130,246,0.2)' : color === 'orange' ? 'rgba(249,115,22,0.2)' : 'rgba(16,185,129,0.2)',
                borderColor: color === 'blue' ? 'rgba(59,130,246,0.3)' : color === 'orange' ? 'rgba(249,115,22,0.3)' : 'rgba(16,185,129,0.3)',
                color: color === 'blue' ? '#60a5fa' : color === 'orange' ? '#fb923c' : '#34d399',
              } : {}}
            >
              {icon}
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>

        {/* Upload Area */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => e.preventDefault()}
          className="relative mb-8"
        >
          <div className="border-2 border-dashed border-white/10 hover:border-white/20 rounded-2xl p-8 sm:p-12 text-center transition-all bg-white/[0.02] hover:bg-white/[0.04]">
            <input
              type="file"
              multiple
              accept="image/*,.raw,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf"
              onChange={handleFileSelect}
              className="hidden"
              id="file-input"
            />
            <label htmlFor="file-input" className="cursor-pointer block">
              <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-white/5 flex items-center justify-center text-gray-500">
                {Icons.upload}
              </div>
              <p className="text-lg font-medium text-white mb-2">
                {mode === 'hdr' ? 'Drop your bracket photos' :
                 mode === 'twilight' ? 'Drop a daytime exterior' : 'Drop photos to enhance'}
              </p>
              <p className="text-sm text-gray-500 mb-4">or click to browse</p>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 text-xs text-gray-400">
                {Icons.photo}
                <span>JPG, PNG, RAW, TIFF supported</span>
              </div>
            </label>
          </div>
        </div>

        {/* Files Selected */}
        {files.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-gray-400">{files.length} file{files.length > 1 ? 's' : ''} selected</span>
              <button onClick={() => setFiles([])} className="text-sm text-red-400 hover:text-red-300">Clear</button>
            </div>

            <div className="grid grid-cols-4 sm:grid-cols-6 gap-2 mb-6">
              {files.slice(0, 12).map((file, i) => (
                <div key={i} className="aspect-video rounded-lg overflow-hidden bg-white/5">
                  {isRawFile(file.name) ? (
                    <div className="w-full h-full flex flex-col items-center justify-center text-gray-500">
                      {Icons.photo}
                      <span className="text-[10px] mt-1">{file.name.split('.').pop()?.toUpperCase()}</span>
                    </div>
                  ) : (
                    <img src={URL.createObjectURL(file)} alt="" className="w-full h-full object-cover" />
                  )}
                </div>
              ))}
              {files.length > 12 && (
                <div className="aspect-video rounded-lg bg-white/5 flex items-center justify-center text-gray-500 text-sm">
                  +{files.length - 12}
                </div>
              )}
            </div>

            {/* Adjustments */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 rounded-xl bg-white/[0.02] border border-white/5 mb-6">
              {[
                { label: 'Brightness', value: brightness, setter: setBrightness },
                { label: 'Contrast', value: contrast, setter: setContrast },
                { label: 'Vibrance', value: vibrance, setter: setVibrance },
                { label: 'Temperature', value: whiteBalance, setter: setWhiteBalance },
              ].map(({ label, value, setter }) => (
                <div key={label}>
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <span>{label}</span>
                    <span>{value > 0 ? '+' : ''}{value.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="-2"
                    max="2"
                    step="0.1"
                    value={value}
                    onChange={(e) => setter(parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-white/10 rounded-full appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              ))}
            </div>

            {/* Enhance Options */}
            {mode === 'enhance' && (
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 mb-6">
                {[
                  { label: 'Window Pull', value: windowPull, setter: setWindowPull },
                  { label: 'Sky Enhance', value: skyEnhance, setter: setSkyEnhance },
                  { label: 'Perspective', value: perspectiveCorrect, setter: setPerspectiveCorrect },
                  { label: 'Denoise', value: noiseReduction, setter: setNoiseReduction },
                  { label: 'Sharpen', value: sharpening, setter: setSharpening },
                ].map(({ label, value, setter }) => (
                  <button
                    key={label}
                    onClick={() => setter(!value)}
                    className={`px-3 py-2 rounded-lg text-xs font-medium transition-all ${
                      value ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' : 'bg-white/5 text-gray-500 border border-white/10'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}

            {/* Progress */}
            {processing && (
              <div className="mb-6">
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">{progressStatus}</span>
                  <span className="text-gray-500">{progress}%</span>
                </div>
                <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all duration-300"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Process Button */}
            <button
              onClick={processImages}
              disabled={processing || files.length === 0}
              className="w-full py-3 sm:py-4 rounded-xl font-semibold text-white bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2"
            >
              {processing ? (
                <>
                  <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Processing...
                </>
              ) : (
                <>
                  {Icons.zap}
                  {mode === 'hdr' ? (files.length === 1 ? 'Enhance Photo' : `Merge ${files.length} Photos`) :
                   mode === 'twilight' ? 'Convert to Dusk' : 'Enhance Photo'}
                </>
              )}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="mb-8 p-4 rounded-xl bg-red-500/10 border border-red-500/20">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Result */}
        {resultUrl && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm font-medium text-green-400 flex items-center gap-2">
                {Icons.check} Processing Complete
              </span>
              <button onClick={resetAll} className="text-sm text-gray-500 hover:text-gray-300">Start Over</button>
            </div>
            <div className="rounded-2xl overflow-hidden bg-white/5 mb-4">
              <img src={resultUrl} alt="Result" className="w-full" />
            </div>
            <button
              onClick={downloadResult}
              className="w-full py-3 rounded-xl font-medium text-white bg-green-600 hover:bg-green-500 transition-all flex items-center justify-center gap-2"
            >
              {Icons.download}
              Download Result
            </button>
          </div>
        )}

        {/* Features */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-16">
          {[
            { title: 'HDR Merge', desc: 'Blend multiple exposures into perfectly balanced images', icon: Icons.layers },
            { title: 'Day to Dusk', desc: 'Transform daytime exteriors into stunning twilight shots', icon: Icons.moon },
            { title: 'AI Enhance', desc: 'Professional corrections with one click', icon: Icons.sparkles },
          ].map(({ title, desc, icon }) => (
            <div key={title} className="p-6 rounded-2xl bg-white/[0.02] border border-white/5 hover:border-white/10 transition-all">
              <div className="w-10 h-10 rounded-xl bg-white/5 flex items-center justify-center text-gray-400 mb-4">
                {icon}
              </div>
              <h3 className="font-semibold text-white mb-2">{title}</h3>
              <p className="text-sm text-gray-500">{desc}</p>
            </div>
          ))}
        </div>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-white/5 text-center">
          <p className="text-sm text-gray-500">
            HDRit · Made by <a href="https://linky.my" className="text-gray-400 hover:text-white transition">Virul</a>
          </p>
          <p className="text-xs text-gray-600 mt-2">{APP_VERSION}</p>
        </footer>
      </div>
    </main>
  )
}
