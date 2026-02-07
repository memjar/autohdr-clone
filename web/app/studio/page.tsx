'use client'

import { useState, useCallback, useRef } from 'react'
import Link from 'next/link'

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

// AI Tools for sidebar
const aiTools = [
  { id: 'hdr', name: 'HDR Editing', icon: '🖼️', desc: 'Blend exposures' },
  { id: 'twilight', name: 'Twilight', icon: '🌙', desc: 'Day to dusk' },
  { id: 'autofill', name: 'AutoFill', icon: '🎨', desc: 'Generative fill' },
  { id: 'remove', name: 'AutoRemove', icon: '🗑️', desc: 'Remove objects' },
  { id: 'grass', name: 'Grass', icon: '🌿', desc: 'Green enhancement' },
  { id: 'staging', name: 'Staging', icon: '🛋️', desc: 'Virtual furniture' },
  { id: 'declutter', name: 'De-clutter', icon: '🧹', desc: 'Clean up' },
  { id: 'repaint', name: 'Re-paint', icon: '🎯', desc: 'Color change' },
]

export default function StudioPage() {
  const [files, setFiles] = useState<File[]>([])
  const [selectedTool, setSelectedTool] = useState('hdr')
  const [processing, setProcessing] = useState(false)
  const [resultUrl, setResultUrl] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'ai' | 'adjustments'>('ai')

  // Adjustments
  const [brightness, setBrightness] = useState(0)
  const [contrast, setContrast] = useState(0)
  const [vibrance, setVibrance] = useState(0)
  const [whiteBalance, setWhiteBalance] = useState(0)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files).filter(f =>
      f.type.startsWith('image/') || RAW_EXTENSIONS.some(ext => f.name.toLowerCase().endsWith(ext))
    )
    setFiles(prev => [...prev, ...droppedFiles])
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const validFiles = Array.from(e.target.files).filter(f =>
        f.type.startsWith('image/') || RAW_EXTENSIONS.some(ext => f.name.toLowerCase().endsWith(ext))
      )
      setFiles(prev => [...prev, ...validFiles])
    }
  }

  const processImages = async () => {
    if (files.length === 0) return
    setProcessing(true)

    try {
      const formData = new FormData()
      files.forEach(file => formData.append('images', file))

      const params = new URLSearchParams({
        mode: selectedTool === 'twilight' ? 'twilight' : 'hdr',
        brightness: brightness.toString(),
        contrast: contrast.toString(),
        vibrance: vibrance.toString(),
        white_balance: whiteBalance.toString(),
      })

      const response = await fetch(`${DEFAULT_BACKEND_URL}/process?${params}`, {
        method: 'POST',
        body: formData,
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        setResultUrl(url)

        // Auto-download
        const a = document.createElement('a')
        a.href = url
        a.download = `hdrit_${selectedTool}_${Date.now()}.jpg`
        document.body.appendChild(a)
        a.click()
        document.body.removeChild(a)
      }
    } catch (err) {
      console.error('Processing error:', err)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="min-h-screen bg-black flex">
      {/* Left Sidebar - AI Tools */}
      <div className="w-64 bg-zinc-900 border-r border-white/10 flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-white/10">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">H</span>
            </div>
            <span className="text-xl font-bold text-white">HDRit Studio</span>
          </Link>
        </div>

        {/* Tool Tabs */}
        <div className="flex border-b border-white/10">
          <button
            onClick={() => setActiveTab('ai')}
            className={`flex-1 py-3 text-sm font-medium transition ${
              activeTab === 'ai' ? 'text-white border-b-2 border-cyan-400' : 'text-gray-400'
            }`}
          >
            AI Tools
          </button>
          <button
            onClick={() => setActiveTab('adjustments')}
            className={`flex-1 py-3 text-sm font-medium transition ${
              activeTab === 'adjustments' ? 'text-white border-b-2 border-cyan-400' : 'text-gray-400'
            }`}
          >
            Adjustments
          </button>
        </div>

        {/* AI Tools List */}
        {activeTab === 'ai' && (
          <div className="flex-1 p-3 space-y-2 overflow-y-auto">
            {aiTools.map(tool => (
              <button
                key={tool.id}
                onClick={() => setSelectedTool(tool.id)}
                className={`w-full flex items-center gap-3 p-3 rounded-lg transition ${
                  selectedTool === tool.id
                    ? 'bg-cyan-500/20 border border-cyan-500/50 text-white'
                    : 'bg-white/5 hover:bg-white/10 text-gray-300'
                }`}
              >
                <span className="text-xl">{tool.icon}</span>
                <div className="text-left">
                  <div className="font-medium text-sm">{tool.name}</div>
                  <div className="text-xs text-gray-500">{tool.desc}</div>
                </div>
              </button>
            ))}
          </div>
        )}

        {/* Adjustments Panel */}
        {activeTab === 'adjustments' && (
          <div className="flex-1 p-4 space-y-4 overflow-y-auto">
            {[
              { label: 'Brightness', value: brightness, setter: setBrightness },
              { label: 'Contrast', value: contrast, setter: setContrast },
              { label: 'Vibrance', value: vibrance, setter: setVibrance },
              { label: 'White Balance', value: whiteBalance, setter: setWhiteBalance },
            ].map(({ label, value, setter }) => (
              <div key={label}>
                <div className="flex justify-between text-xs text-gray-400 mb-1">
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
                  className="w-full h-2 bg-white/20 rounded-full appearance-none cursor-pointer accent-cyan-400"
                />
              </div>
            ))}
            <div className="flex gap-2 pt-4">
              <button
                onClick={() => {
                  setBrightness(0)
                  setContrast(0)
                  setVibrance(0)
                  setWhiteBalance(0)
                }}
                className="flex-1 py-2 text-sm text-gray-400 hover:text-white bg-white/5 rounded-lg"
              >
                Reset
              </button>
              <button
                onClick={processImages}
                disabled={files.length === 0 || processing}
                className="flex-1 py-2 text-sm font-medium text-black bg-cyan-400 hover:bg-cyan-300 rounded-lg disabled:opacity-50"
              >
                Apply
              </button>
            </div>
          </div>
        )}

        {/* Process Button */}
        <div className="p-4 border-t border-white/10">
          <button
            onClick={processImages}
            disabled={files.length === 0 || processing}
            className="w-full py-3 rounded-lg font-semibold text-black bg-cyan-400 hover:bg-cyan-300 disabled:opacity-50 disabled:cursor-not-allowed transition flex items-center justify-center gap-2"
          >
            {processing ? (
              <>
                <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                Processing...
              </>
            ) : (
              <>Process {files.length > 0 ? `${files.length} Photo${files.length > 1 ? 's' : ''}` : 'Photos'}</>
            )}
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="h-14 bg-zinc-900/50 border-b border-white/10 flex items-center justify-between px-4">
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">
              {files.length > 0 ? `${files.length} file${files.length > 1 ? 's' : ''} selected` : 'No files selected'}
            </span>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition">
              Show Inputs
            </button>
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition">
              Report Feedback
            </button>
          </div>
        </div>

        {/* Upload / Preview Area */}
        <div className="flex-1 flex items-center justify-center p-8">
          {files.length === 0 ? (
            <div
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
              className="w-full max-w-2xl"
            >
              <div className="border-2 border-dashed border-white/20 hover:border-cyan-400/50 rounded-2xl p-12 text-center transition-all bg-white/[0.02] hover:bg-white/[0.05]">
                <input
                  type="file"
                  multiple
                  accept="image/*,.raw,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="studio-file-input"
                />
                <label htmlFor="studio-file-input" className="cursor-pointer block">
                  <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-white/10 flex items-center justify-center">
                    <svg className="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                  <p className="text-xl font-semibold text-white mb-2">
                    Drag & drop your images here, or
                  </p>
                  <button className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition mb-4">
                    Browse Files
                  </button>

                  {/* Cloud Import Options */}
                  <div className="flex items-center justify-center gap-4 mt-6 pt-6 border-t border-white/10">
                    <button
                      onClick={(e) => { e.preventDefault(); alert('Dropbox integration coming soon!') }}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all"
                    >
                      <svg className="w-5 h-5 text-blue-400" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M6 2l6 3.75L6 9.5 0 5.75 6 2zm12 0l6 3.75-6 3.75-6-3.75L18 2zM0 13.25L6 9.5l6 3.75-6 3.75-6-3.75zm18-3.75l6 3.75-6 3.75-6-3.75 6-3.75zM6 17.5l6-3.75 6 3.75-6 3.75-6-3.75z"/>
                      </svg>
                      <span className="text-sm text-gray-300">Select from Dropbox</span>
                    </button>
                    <button
                      onClick={(e) => { e.preventDefault(); alert('Google Drive integration coming soon!') }}
                      className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 transition-all"
                    >
                      <svg className="w-5 h-5" viewBox="0 0 24 24">
                        <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                        <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                        <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                        <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
                      </svg>
                      <span className="text-sm text-gray-300">Select from Google Drive</span>
                    </button>
                  </div>
                </label>
              </div>
            </div>
          ) : resultUrl ? (
            <div className="max-w-4xl w-full">
              <img src={resultUrl} alt="Processed result" className="w-full rounded-xl" />
              <div className="flex gap-4 mt-6 justify-center">
                <button
                  onClick={() => { setFiles([]); setResultUrl(null) }}
                  className="px-6 py-3 bg-white/10 text-white rounded-lg hover:bg-white/20 transition"
                >
                  New Edit
                </button>
                <a
                  href={resultUrl}
                  download={`hdrit_${selectedTool}_${Date.now()}.jpg`}
                  className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition"
                >
                  Download
                </a>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-4 gap-4 max-w-4xl">
              {files.map((file, i) => (
                <div key={i} className="aspect-square rounded-lg bg-white/5 border border-white/10 overflow-hidden relative group">
                  {!isRawFile(file.name) && file.type.startsWith('image/') ? (
                    <img
                      src={URL.createObjectURL(file)}
                      alt={file.name}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex flex-col items-center justify-center text-gray-400">
                      <span className="text-xs font-bold text-cyan-400">{file.name.split('.').pop()?.toUpperCase()}</span>
                    </div>
                  )}
                  <button
                    onClick={() => setFiles(files.filter((_, idx) => idx !== i))}
                    className="absolute top-2 right-2 w-6 h-6 bg-red-500 hover:bg-red-400 rounded-full flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <svg className="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
              ))}
              <label
                htmlFor="studio-file-input-add"
                className="aspect-square rounded-lg bg-white/5 border border-dashed border-white/20 hover:border-cyan-400/50 flex items-center justify-center cursor-pointer transition"
              >
                <input
                  type="file"
                  multiple
                  accept="image/*,.raw,.cr2,.cr3,.nef,.arw,.dng,.orf,.rw2,.pef,.raf"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="studio-file-input-add"
                />
                <svg className="w-8 h-8 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
                </svg>
              </label>
            </div>
          )}
        </div>

        {/* Quality Feedback */}
        {resultUrl && (
          <div className="p-4 bg-zinc-900/50 border-t border-white/10">
            <div className="max-w-md mx-auto text-center">
              <p className="text-sm text-gray-400 mb-3">How does this edit look? (1 = worst, 10 = best)</p>
              <div className="flex gap-2 justify-center">
                {[1,2,3,4,5,6,7,8,9,10].map(n => (
                  <button
                    key={n}
                    className="w-8 h-8 rounded-lg bg-white/10 hover:bg-cyan-400 hover:text-black text-gray-400 text-sm font-medium transition"
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
