'use client'

import { useState, useCallback } from 'react'

export default function Home() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)
  const [result, setResult] = useState<string | null>(null)
  const [mode, setMode] = useState<'hdr' | 'twilight'>('hdr')

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFiles = Array.from(e.dataTransfer.files).filter(f =>
      f.type.startsWith('image/')
    )
    setFiles(prev => [...prev, ...droppedFiles])
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(prev => [...prev, ...Array.from(e.target.files!)])
    }
  }

  const processImages = async () => {
    if (files.length === 0) return

    setProcessing(true)
    setResult(null)

    // TODO: Connect to backend API
    // For now, simulate processing
    await new Promise(r => setTimeout(r, 2000))

    setProcessing(false)
    setResult('Processing complete! (Backend not connected yet)')
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="text-center mb-12">
        <h1 className="text-4xl font-bold mb-2">AutoHDR Clone</h1>
        <p className="text-gray-400">Open-source AI real estate photo editing</p>
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
          accept="image/*"
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
              <div key={i} className="aspect-video bg-gray-800 rounded-lg overflow-hidden">
                <img
                  src={URL.createObjectURL(file)}
                  alt={file.name}
                  className="w-full h-full object-cover"
                />
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

      {/* Result */}
      {result && (
        <div className="mt-8 p-6 bg-green-900/30 border border-green-700 rounded-lg">
          <p className="text-green-400">{result}</p>
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
