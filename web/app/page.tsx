'use client'

import { useState, useCallback, useRef, useEffect } from 'react'
import {
  SignInButton,
  SignUpButton,
  SignedIn,
  SignedOut,
  UserButton,
} from '@clerk/nextjs'

// Check if Clerk is configured
const isClerkConfigured = typeof window !== 'undefined'
  ? !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY?.startsWith('pk_')
  : !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY?.startsWith('pk_')

const APP_VERSION = 'v3.2.0'
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

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [originalUrl, setOriginalUrl] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mode, setMode] = useState<'hdr' | 'twilight' | 'enhance' | 'grass' | 'staging' | 'removal'>('hdr')
  const [backendUrl] = useState(DEFAULT_BACKEND_URL)
  const [proProcessorStatus, setProProcessorStatus] = useState<'checking' | 'connected' | 'unavailable'>('checking')
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
  const [uploadProgress, setUploadProgress] = useState(0)
  const [isUploading, setIsUploading] = useState(false)
  const progressInterval = useRef<NodeJS.Timeout | null>(null)
  const [previewUrls, setPreviewUrls] = useState<string[]>([])

  // Generate preview URLs when files change
  useEffect(() => {
    // Revoke old URLs to prevent memory leaks
    previewUrls.forEach(url => URL.revokeObjectURL(url))

    // Create new preview URLs for non-RAW files
    const newUrls = files.map(file => {
      if (!isRawFile(file.name) && file.type.startsWith('image/')) {
        return URL.createObjectURL(file)
      }
      return ''
    })
    setPreviewUrls(newUrls)

    // Cleanup on unmount
    return () => {
      newUrls.forEach(url => url && URL.revokeObjectURL(url))
    }
  }, [files])

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
          } else {
            setProProcessorStatus('unavailable')
          }
        } else {
          setProProcessorStatus('unavailable')
        }
      } catch {
        setProProcessorStatus('unavailable')
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

    setProcessing(true)
    setIsUploading(true)
    setResult(null)
    setResultUrl(null)
    setOriginalUrl(null)
    setError(null)
    setProgress(0)
    setUploadProgress(0)
    setProgressStatus('Preparing upload...')

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

      const apiUrl = `/api/process?${params}`

      // Calculate total file size for progress display
      const totalSize = files.reduce((acc, f) => acc + f.size, 0)
      const totalSizeMB = (totalSize / 1024 / 1024).toFixed(1)

      // Use XMLHttpRequest for upload progress
      console.log('Starting upload to:', apiUrl)
      console.log('Total size:', totalSizeMB, 'MB')

      const blob = await new Promise<Blob>((resolve, reject) => {
        const xhr = new XMLHttpRequest()

        xhr.upload.addEventListener('progress', (e) => {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100)
            setUploadProgress(pct)
            setProgress(Math.round(pct * 0.5)) // Upload is 50% of total progress
            const uploadedMB = (e.loaded / 1024 / 1024).toFixed(1)
            setProgressStatus(`Uploading... ${uploadedMB}MB / ${totalSizeMB}MB`)
          }
        })

        xhr.upload.addEventListener('load', () => {
          setIsUploading(false)
          setProgress(50)
          setProgressStatus('Processing on server...')

          // Start processing progress simulation
          const stages = mode === 'enhance' ? [
            { pct: 55, msg: 'Analyzing image...' },
            { pct: 65, msg: 'Correcting exposure...' },
            { pct: 75, msg: 'Balancing colors...' },
            { pct: 85, msg: 'Enhancing details...' },
            { pct: 92, msg: 'Finalizing...' },
          ] : mode === 'twilight' ? [
            { pct: 60, msg: 'Analyzing scene...' },
            { pct: 75, msg: 'Generating sky...' },
            { pct: 88, msg: 'Blending lights...' },
          ] : [
            { pct: 55, msg: 'Denoising...' },
            { pct: 65, msg: 'HDR fusion...' },
            { pct: 75, msg: 'Tone mapping...' },
            { pct: 85, msg: 'Color correction...' },
            { pct: 92, msg: 'Finalizing...' },
          ]

          let stageIndex = 0
          progressInterval.current = setInterval(() => {
            if (stageIndex < stages.length) {
              setProgress(stages[stageIndex].pct)
              setProgressStatus(stages[stageIndex].msg)
              stageIndex++
            }
          }, 1200)
        })

        xhr.addEventListener('load', () => {
          if (progressInterval.current) {
            clearInterval(progressInterval.current)
            progressInterval.current = null
          }

          console.log('XHR load complete, status:', xhr.status, 'response size:', xhr.response?.size)

          if (xhr.status === 200 && xhr.response) {
            resolve(xhr.response)
          } else {
            console.error('XHR error response:', xhr.responseText || xhr.response)
            reject(new Error(`Processing failed: ${xhr.status}`))
          }
        })

        xhr.addEventListener('error', () => {
          reject(new Error('Network error - check your connection'))
        })

        xhr.addEventListener('timeout', () => {
          reject(new Error('Request timed out'))
        })

        xhr.responseType = 'blob'
        xhr.timeout = 300000 // 5 minutes
        xhr.open('POST', apiUrl)
        xhr.send(formData)
      })

      console.log('Received blob:', blob.size, 'bytes, type:', blob.type)

      if (blob.size < 1000) {
        console.error('Blob too small:', blob.size)
        throw new Error('Invalid response from server')
      }

      const url = URL.createObjectURL(blob)
      console.log('Created blob URL:', url)

      setProgress(100)
      setProgressStatus('Complete!')
      setResultUrl(url)
      setResult('Processing complete!')

      // Auto-download result
      console.log('Triggering download...')
      const a = document.createElement('a')
      a.href = url
      a.download = `hdrit_${mode}_${Date.now()}.jpg`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      console.log('Download triggered')
    } catch (err: any) {
      if (err.message.includes('timeout')) {
        setError('Request timed out. Check if the backend is accessible.')
      } else if (err.message.includes('Network') || err.message.includes('fetch')) {
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
      setIsUploading(false)
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

  const services = [
    { id: 'hdr', name: 'HDR Editing', desc: 'Blend multiple exposures perfectly', icon: '🏠' },
    { id: 'enhance', name: 'Flambient Editing', desc: 'Flash + ambient blend', icon: '💡' },
    { id: 'twilight', name: 'Day to Dusk', desc: 'Transform to twilight', icon: '🌅' },
    { id: 'grass', name: 'Grass Greening', desc: 'Enhance lawn color', icon: '🌿' },
    { id: 'removal', name: 'Item Removal', desc: 'Remove unwanted objects', icon: '✨' },
    { id: 'staging', name: 'Virtual Staging', desc: 'Add virtual furniture', icon: '🛋️' },
  ]

  return (
    <main className="min-h-screen bg-black">
      {/* Navigation - AutoHDR Style */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm border-b border-white/5">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <a href="/" className="flex items-center gap-2">
              <span className="text-2xl font-bold text-white">HDRit</span>
            </a>

            {/* Desktop Nav */}
            <div className="hidden md:flex items-center gap-8">
              <a href="/pricing" className="text-sm text-gray-300 hover:text-white transition">Pricing</a>
              <a href="/dashboard" className="text-sm text-gray-300 hover:text-white transition">Studio</a>
              <a href="/about" className="text-sm text-gray-300 hover:text-white transition">About</a>
              <a href="mailto:support@hdr.it.com" className="text-sm text-gray-300 hover:text-white transition">Contact</a>
            </div>

            {/* Auth Buttons */}
            <div className="hidden md:flex items-center gap-4">
              {isClerkConfigured ? (
                <>
                  <SignedIn>
                    <a href="/dashboard" className="text-sm text-gray-300 hover:text-white transition">Dashboard</a>
                    <UserButton
                      appearance={{
                        elements: {
                          avatarBox: 'w-8 h-8',
                        },
                      }}
                    />
                  </SignedIn>
                  <SignedOut>
                    <SignInButton mode="modal">
                      <button className="text-sm text-gray-300 hover:text-white transition">
                        Sign In
                      </button>
                    </SignInButton>
                    <SignUpButton mode="modal">
                      <button className="px-4 py-2 text-sm font-medium text-black bg-white hover:bg-gray-100 rounded-lg transition">
                        Get Started
                      </button>
                    </SignUpButton>
                  </SignedOut>
                </>
              ) : (
                <>
                  <a href="/sign-in" className="text-sm text-gray-300 hover:text-white transition">Sign In</a>
                  <a href="/sign-up" className="px-4 py-2 text-sm font-medium text-black bg-white hover:bg-gray-100 rounded-lg transition">
                    Get Started
                  </a>
                </>
              )}
            </div>

            {/* Mobile Menu Button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-gray-400 hover:text-white"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden bg-black border-t border-white/5">
            <div className="px-4 py-4 space-y-3">
              <a href="/pricing" className="block text-gray-300 hover:text-white">Pricing</a>
              <a href="/dashboard" className="block text-gray-300 hover:text-white">Studio</a>
              <a href="/about" className="block text-gray-300 hover:text-white">About</a>
              <a href="mailto:support@hdr.it.com" className="block text-gray-300 hover:text-white">Contact</a>
              <div className="pt-3 border-t border-white/10">
                {isClerkConfigured ? (
                  <>
                    <SignedOut>
                      <SignInButton mode="modal">
                        <button className="block w-full text-left text-gray-300 hover:text-white mb-2">Sign In</button>
                      </SignInButton>
                      <SignUpButton mode="modal">
                        <button className="block w-full px-4 py-2 text-sm font-medium text-black bg-white rounded-lg">
                          Get Started
                        </button>
                      </SignUpButton>
                    </SignedOut>
                    <SignedIn>
                      <a href="/dashboard" className="block text-gray-300 hover:text-white mb-2">Dashboard</a>
                    </SignedIn>
                  </>
                ) : (
                  <>
                    <a href="/sign-in" className="block text-gray-300 hover:text-white mb-2">Sign In</a>
                    <a href="/sign-up" className="block w-full px-4 py-2 text-sm font-medium text-center text-black bg-white rounded-lg">
                      Get Started
                    </a>
                  </>
                )}
              </div>
            </div>
          </div>
        )}
      </nav>

      {/* Hero Section - AutoHDR Style */}
      <section className="pt-32 pb-16 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-6 leading-tight">
            AI Photo Editing for
            <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
              Real Estate Photographers
            </span>
          </h1>
          <p className="text-xl text-gray-400 mb-12 max-w-2xl mx-auto">
            Edit your photoshoot in minutes at half the cost of an editor.
            Professional results, powered by AI.
          </p>

          {/* Status Indicator */}
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 mb-8">
            <div className={`w-2 h-2 rounded-full ${
              proProcessorStatus === 'connected' ? 'bg-green-500' :
              proProcessorStatus === 'checking' ? 'bg-yellow-500 animate-pulse' : 'bg-gray-500'
            }`} />
            <span className="text-sm text-gray-400">
              {proProcessorStatus === 'connected' ? 'Pro Engine Ready' :
               proProcessorStatus === 'checking' ? 'Connecting...' : 'Cloud Processing'}
            </span>
          </div>
        </div>
      </section>

      {/* Services Grid - AutoHDR Style */}
      <section className="py-8 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {services.map((service) => (
              <button
                key={service.id}
                onClick={() => setMode(service.id as typeof mode)}
                className={`p-4 rounded-xl text-center transition-all ${
                  mode === service.id
                    ? 'bg-white text-black'
                    : 'bg-white/5 text-white hover:bg-white/10 border border-white/10'
                }`}
              >
                <div className="text-2xl mb-2">{service.icon}</div>
                <div className="text-sm font-medium">{service.name}</div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Upload Section - AutoHDR Style */}
      <section className="py-8 px-4">
        <div className="max-w-3xl mx-auto">
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            className="relative"
          >
            <div className="border-2 border-dashed border-white/20 hover:border-white/40 rounded-2xl p-12 text-center transition-all bg-white/[0.02] hover:bg-white/[0.05]">
              <input
                type="file"
                multiple
                accept="image/*,.raw,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf"
                onChange={handleFileSelect}
                className="hidden"
                id="file-input"
              />
              <label htmlFor="file-input" className="cursor-pointer block">
                <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-white/10 flex items-center justify-center">
                  <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                </div>
                <p className="text-xl font-semibold text-white mb-2">
                  Upload your photos
                </p>
                <p className="text-gray-500 mb-6">
                  Drag & drop or click to browse
                </p>
                <button className="px-8 py-3 bg-white text-black font-semibold rounded-lg hover:bg-gray-100 transition">
                  Select Files
                </button>
                <p className="text-xs text-gray-600 mt-4">
                  Supports JPG, PNG, RAW, TIFF, HEIC
                </p>
              </label>
            </div>
          </div>

          {/* Files Selected */}
          {files.length > 0 && (
            <div className="mt-8">
              <div className="flex items-center justify-between mb-4">
                <span className="text-white font-medium">{files.length} file{files.length > 1 ? 's' : ''} selected</span>
                <button onClick={() => setFiles([])} className="text-sm text-red-400 hover:text-red-300">Clear all</button>
              </div>

              {/* Thumbnail Grid */}
              <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-3 mb-6">
                {files.slice(0, 12).map((file, i) => (
                  <div key={i} className="group relative">
                    <div className="aspect-square rounded-lg overflow-hidden bg-white/5 border border-white/10 group-hover:border-white/30 transition-all">
                      {isRawFile(file.name) ? (
                        <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 bg-gradient-to-br from-white/5 to-transparent">
                          <span className="text-xs font-medium text-blue-400">{file.name.split('.').pop()?.toUpperCase()}</span>
                        </div>
                      ) : previewUrls[i] ? (
                        <img
                          src={previewUrls[i]}
                          alt={file.name}
                          loading="lazy"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                        </div>
                      )}
                    </div>
                    <p className="mt-1 text-[10px] text-gray-500 truncate" title={file.name}>
                      {file.name}
                    </p>
                    {/* Remove button */}
                    <button
                      onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                      className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 hover:bg-red-400 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                      <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ))}
                {files.length > 12 && (
                  <div className="aspect-square rounded-lg bg-white/5 border border-white/10 flex flex-col items-center justify-center text-gray-400">
                    <span className="text-xl font-bold">+{files.length - 12}</span>
                    <span className="text-[10px]">more</span>
                  </div>
                )}
              </div>

              {/* Adjustments Panel */}
              <div className="p-5 rounded-xl bg-white/5 border border-white/10 mb-6">
                <h3 className="text-sm font-medium text-white mb-4">Adjustments</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  {[
                    { label: 'Brightness', value: brightness, setter: setBrightness },
                    { label: 'Contrast', value: contrast, setter: setContrast },
                    { label: 'Vibrance', value: vibrance, setter: setVibrance },
                    { label: 'Temperature', value: whiteBalance, setter: setWhiteBalance },
                  ].map(({ label, value, setter }) => (
                    <div key={label}>
                      <div className="flex justify-between text-xs text-gray-400 mb-2">
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
                        className="w-full h-1 bg-white/20 rounded-full appearance-none cursor-pointer accent-white"
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Enhance Options */}
              {mode === 'enhance' && (
                <div className="flex flex-wrap gap-2 mb-6">
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
                      className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                        value ? 'bg-white text-black' : 'bg-white/10 text-gray-400 hover:bg-white/20'
                      }`}
                    >
                      {value && <span className="mr-1">✓</span>}
                      {label}
                    </button>
                  ))}
                </div>
              )}

              {/* Progress */}
              {processing && (
                <div className="mb-6">
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-white flex items-center gap-2">
                      {isUploading ? (
                        <svg className="w-4 h-4 animate-pulse text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                        </svg>
                      ) : (
                        <svg className="w-4 h-4 animate-spin text-green-400" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                      )}
                      {progressStatus}
                    </span>
                    <span className="text-gray-400">{progress}%</span>
                  </div>
                  <div className="h-3 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        isUploading ? 'bg-blue-500' : 'bg-green-500'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-2">
                    {isUploading ? 'Please wait while your files upload...' : 'Processing with Bulletproof v6.0.8...'}
                  </p>
                </div>
              )}

              {/* Process Button */}
              <button
                onClick={processImages}
                disabled={processing || files.length === 0}
                className={`w-full py-4 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 ${
                  processing
                    ? isUploading
                      ? 'bg-blue-500 text-white cursor-wait'
                      : 'bg-green-500 text-white cursor-wait'
                    : 'bg-white text-black hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed'
                }`}
              >
                {processing ? (
                  isUploading ? (
                    <>
                      <svg className="w-5 h-5 animate-bounce" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                      </svg>
                      Uploading {uploadProgress}%...
                    </>
                  ) : (
                    <>
                      <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Processing...
                    </>
                  )
                ) : (
                  <>
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Process {files.length} Photo{files.length > 1 ? 's' : ''}
                  </>
                )}
              </button>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30">
              <p className="text-red-400 text-sm">{error}</p>
            </div>
          )}

          {/* Result */}
          {resultUrl && (
            <div className="mt-8">
              <div className="flex items-center justify-between mb-4">
                <span className="text-white font-medium flex items-center gap-2">
                  <svg className="w-5 h-5 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Processing Complete
                </span>
                <button onClick={resetAll} className="text-sm text-gray-400 hover:text-white">Start Over</button>
              </div>
              <div className="rounded-2xl overflow-hidden bg-white/5 mb-4">
                <img src={resultUrl} alt="Result" className="w-full" />
              </div>
              <button
                onClick={downloadResult}
                className="w-full py-4 rounded-xl font-semibold text-black bg-green-500 hover:bg-green-400 transition-all flex items-center justify-center gap-2"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Download Result
              </button>
            </div>
          )}
        </div>
      </section>

      {/* Examples Section - Before/After Showcase */}
      <section className="py-20 px-4 bg-zinc-950">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-4">
            See the Difference
          </h2>
          <p className="text-gray-400 text-center mb-12 max-w-2xl mx-auto">
            From single photo edits to batch processing hundreds of images
          </p>

          {/* Single Photo Example */}
          <div className="mb-16">
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-sm">1</span>
              Single Photo Edit
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              {/* Before/After HDR */}
              <div className="rounded-2xl overflow-hidden border border-white/10">
                <div className="relative aspect-[4/3] bg-zinc-900">
                  {/* Replace with actual before image */}
                  <img
                    src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=800&q=80"
                    alt="Before HDR"
                    className="w-full h-full object-cover brightness-75 contrast-125"
                  />
                  <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-black/50 text-white text-xs font-medium">
                    Before
                  </div>
                </div>
                <div className="p-4 bg-zinc-900/50">
                  <p className="text-sm text-gray-400">Original RAW bracket - dark windows, blown highlights</p>
                </div>
              </div>
              <div className="rounded-2xl overflow-hidden border border-white/10">
                <div className="relative aspect-[4/3] bg-zinc-900">
                  {/* Replace with actual after image */}
                  <img
                    src="https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=800&q=80"
                    alt="After HDR"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-green-500/80 text-white text-xs font-medium">
                    After
                  </div>
                </div>
                <div className="p-4 bg-zinc-900/50">
                  <p className="text-sm text-gray-400">HDR merged - balanced exposure, natural colors</p>
                </div>
              </div>
            </div>
          </div>

          {/* Day to Dusk Example */}
          <div className="mb-16">
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-orange-500/20 text-orange-400 flex items-center justify-center text-sm">2</span>
              Day to Dusk Transformation
            </h3>
            <div className="grid md:grid-cols-2 gap-6">
              <div className="rounded-2xl overflow-hidden border border-white/10">
                <div className="relative aspect-[4/3] bg-zinc-900">
                  <img
                    src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800&q=80"
                    alt="Day exterior"
                    className="w-full h-full object-cover"
                  />
                  <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-black/50 text-white text-xs font-medium">
                    Day
                  </div>
                </div>
                <div className="p-4 bg-zinc-900/50">
                  <p className="text-sm text-gray-400">Midday exterior - harsh shadows, bland sky</p>
                </div>
              </div>
              <div className="rounded-2xl overflow-hidden border border-white/10">
                <div className="relative aspect-[4/3] bg-zinc-900">
                  <img
                    src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?w=800&q=80"
                    alt="Dusk exterior"
                    className="w-full h-full object-cover"
                    style={{ filter: 'sepia(0.3) saturate(1.5) hue-rotate(-20deg)' }}
                  />
                  <div className="absolute top-4 left-4 px-3 py-1 rounded-full bg-orange-500/80 text-white text-xs font-medium">
                    Dusk
                  </div>
                </div>
                <div className="p-4 bg-zinc-900/50">
                  <p className="text-sm text-gray-400">Twilight conversion - warm glow, dramatic sky</p>
                </div>
              </div>
            </div>
          </div>

          {/* Batch Processing Example */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <span className="w-8 h-8 rounded-full bg-purple-500/20 text-purple-400 flex items-center justify-center text-sm">3</span>
              Batch Processing - Full Property Shoot
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
              {[
                'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=400&q=80',
                'https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?w=400&q=80',
                'https://images.unsplash.com/photo-1600573472592-401b489a3cdc?w=400&q=80',
                'https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=400&q=80',
                'https://images.unsplash.com/photo-1600585154526-990dced4db0d?w=400&q=80',
                'https://images.unsplash.com/photo-1600566753086-00f18fb6b3ea?w=400&q=80',
              ].map((src, i) => (
                <div key={i} className="relative aspect-[4/3] rounded-xl overflow-hidden border border-white/10 group">
                  <img
                    src={src}
                    alt={`Property photo ${i + 1}`}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="absolute bottom-2 left-2 right-2">
                      <div className="flex items-center gap-1">
                        <svg className="w-3 h-3 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                        </svg>
                        <span className="text-[10px] text-white">Edited</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 p-4 rounded-xl bg-white/5 border border-white/10">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <p className="text-white font-medium">42 photos processed</p>
                  <p className="text-sm text-gray-500">Complete property shoot - HDR merged & color corrected</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-white">3:42</p>
                    <p className="text-xs text-gray-500">Total time</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-green-400">$8.40</p>
                    <p className="text-xs text-gray-500">Cost</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section - AutoHDR Style */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl font-bold text-white text-center mb-4">
            Everything you need for real estate photos
          </h2>
          <p className="text-gray-400 text-center mb-12 max-w-2xl mx-auto">
            Professional editing tools powered by AI, delivered in minutes instead of days
          </p>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: 'HDR Editing',
                desc: 'Blend multiple exposures into perfectly balanced images with natural lighting and detail in every corner.',
                icon: '🏠'
              },
              {
                title: 'Day to Dusk',
                desc: 'Transform daytime exteriors into stunning twilight shots with warm interior lights and dramatic skies.',
                icon: '🌅'
              },
              {
                title: 'Virtual Staging',
                desc: 'Add beautiful virtual furniture to empty rooms. Make vacant properties feel like home.',
                icon: '🛋️'
              },
              {
                title: 'Grass Greening',
                desc: 'Make lawns look lush and healthy. Perfect for listings with dormant or patchy grass.',
                icon: '🌿'
              },
              {
                title: 'Item Removal',
                desc: 'Remove unwanted objects, vehicles, or clutter. Clean up your photos without reshooting.',
                icon: '✨'
              },
              {
                title: 'Sky Replacement',
                desc: 'Replace overcast skies with beautiful blue skies and perfect clouds automatically.',
                icon: '☁️'
              },
            ].map((feature) => (
              <div key={feature.title} className="p-6 rounded-2xl bg-white/[0.03] border border-white/10 hover:border-white/20 transition-all">
                <div className="text-3xl mb-4">{feature.icon}</div>
                <h3 className="text-lg font-semibold text-white mb-2">{feature.title}</h3>
                <p className="text-sm text-gray-400">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 border-t border-white/5">
        <div className="max-w-3xl mx-auto text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to transform your photos?
          </h2>
          <p className="text-gray-400 mb-8">
            Join thousands of real estate photographers who trust HDRit for their editing needs.
          </p>
          {isClerkConfigured ? (
            <SignedOut>
              <SignUpButton mode="modal">
                <button className="px-8 py-4 text-lg font-semibold text-black bg-white hover:bg-gray-100 rounded-xl transition">
                  Get Started Free
                </button>
              </SignUpButton>
            </SignedOut>
          ) : (
            <a href="/sign-up" className="inline-block px-8 py-4 text-lg font-semibold text-black bg-white hover:bg-gray-100 rounded-xl transition">
              Get Started Free
            </a>
          )}
          <p className="text-sm text-gray-500 mt-4">No credit card required</p>
        </div>
      </section>

      {/* Footer - AutoHDR Style */}
      <footer className="py-12 px-4 border-t border-white/5 bg-black">
        <div className="max-w-6xl mx-auto">
          <div className="grid md:grid-cols-4 gap-8 mb-12">
            {/* Brand */}
            <div>
              <h3 className="text-xl font-bold text-white mb-4">HDRit</h3>
              <p className="text-sm text-gray-500 mb-4">
                AI-powered photo editing for real estate professionals.
              </p>
              <p className="text-sm text-gray-600">
                Made by <a href="https://virul.co" className="text-gray-400 hover:text-white transition">Virul</a>
              </p>
            </div>

            {/* Product */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-4">Product</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="/pricing" className="text-gray-500 hover:text-white transition">Pricing</a></li>
                <li><a href="/dashboard" className="text-gray-500 hover:text-white transition">Studio</a></li>
                <li><a href="/about" className="text-gray-500 hover:text-white transition">About</a></li>
              </ul>
            </div>

            {/* Services */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-4">Services</h4>
              <ul className="space-y-2 text-sm">
                <li><span className="text-gray-500">HDR Editing</span></li>
                <li><span className="text-gray-500">Day to Dusk</span></li>
                <li><span className="text-gray-500">Virtual Staging</span></li>
                <li><span className="text-gray-500">Item Removal</span></li>
              </ul>
            </div>

            {/* Contact */}
            <div>
              <h4 className="text-sm font-semibold text-white mb-4">Contact</h4>
              <ul className="space-y-2 text-sm">
                <li><a href="mailto:support@hdr.it.com" className="text-gray-500 hover:text-white transition">support@hdr.it.com</a></li>
                <li><span className="text-gray-500">Toronto, Canada</span></li>
              </ul>
            </div>
          </div>

          <div className="pt-8 border-t border-white/5 flex flex-col sm:flex-row justify-between items-center gap-4">
            <p className="text-xs text-gray-600">
              © 2026 HDRit. All rights reserved.
            </p>
            <p className="text-xs text-gray-700">{APP_VERSION}</p>
          </div>
        </div>
      </footer>
    </main>
  )
}
