'use client'

import { useState } from 'react'
import Link from 'next/link'

// Mock data for demo
const mockProjects = [
  { id: '1', name: '123 Main St', photos: 12, date: '2026-02-07', status: 'completed' },
  { id: '2', name: '456 Oak Ave', photos: 8, date: '2026-02-06', status: 'completed' },
  { id: '3', name: '789 Pine Rd', photos: 15, date: '2026-02-05', status: 'processing' },
]

const mockPhotos = [
  { id: '1', name: 'IMG_001.jpg', url: 'https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?w=400&q=80' },
  { id: '2', name: 'IMG_002.jpg', url: 'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?w=400&q=80' },
  { id: '3', name: 'IMG_003.jpg', url: 'https://images.unsplash.com/photo-1600566753190-17f0baa2a6c3?w=400&q=80' },
  { id: '4', name: 'IMG_004.jpg', url: 'https://images.unsplash.com/photo-1600573472592-401b489a3cdc?w=400&q=80' },
  { id: '5', name: 'IMG_005.jpg', url: 'https://images.unsplash.com/photo-1600047509807-ba8f99d2cdde?w=400&q=80' },
  { id: '6', name: 'IMG_006.jpg', url: 'https://images.unsplash.com/photo-1600585154526-990dced4db0d?w=400&q=80' },
]

export default function ListingsPage() {
  const [selectedProject, setSelectedProject] = useState(mockProjects[0])
  const [selectedPhotos, setSelectedPhotos] = useState<string[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [editingProject, setEditingProject] = useState<string | null>(null)

  const togglePhotoSelection = (id: string) => {
    setSelectedPhotos(prev =>
      prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]
    )
  }

  const selectAll = () => {
    if (selectedPhotos.length === mockPhotos.length) {
      setSelectedPhotos([])
    } else {
      setSelectedPhotos(mockPhotos.map(p => p.id))
    }
  }

  return (
    <div className="min-h-screen bg-black flex">
      {/* Left Sidebar - Projects */}
      <div className="w-64 bg-zinc-900 border-r border-white/10 flex flex-col">
        {/* Logo */}
        <div className="p-4 border-b border-white/10">
          <Link href="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">H</span>
            </div>
            <span className="text-xl font-bold text-white">Listings</span>
          </Link>
        </div>

        {/* Project List */}
        <div className="flex-1 overflow-y-auto p-2">
          {mockProjects.map(project => (
            <div
              key={project.id}
              onClick={() => setSelectedProject(project)}
              className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer transition mb-1 ${
                selectedProject.id === project.id
                  ? 'bg-cyan-500/20 border border-cyan-500/50'
                  : 'hover:bg-white/5'
              }`}
            >
              <div className="flex items-center gap-3">
                <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
                </svg>
                {editingProject === project.id ? (
                  <input
                    type="text"
                    defaultValue={project.name}
                    className="bg-transparent text-white text-sm border-b border-cyan-400 focus:outline-none"
                    onBlur={() => setEditingProject(null)}
                    onKeyDown={(e) => e.key === 'Enter' && setEditingProject(null)}
                    autoFocus
                  />
                ) : (
                  <span className="text-sm text-white">{project.name}</span>
                )}
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); setEditingProject(project.id) }}
                className="opacity-0 group-hover:opacity-100 p-1 hover:bg-white/10 rounded transition"
              >
                <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
            </div>
          ))}
        </div>

        {/* New Project Button */}
        <div className="p-4 border-t border-white/10">
          <button className="w-full py-2 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition flex items-center justify-center gap-2">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            New Project
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="h-14 bg-zinc-900/50 border-b border-white/10 flex items-center justify-between px-4">
          {/* Search */}
          <div className="relative">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search listings..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-64 pl-10 pr-4 py-2 bg-white/5 border border-white/10 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400/50"
            />
          </div>

          {/* Actions */}
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
              Copy Link
            </button>
            <button className="px-3 py-1.5 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
              </svg>
              Rate
            </button>
            <button
              onClick={selectAll}
              className="px-3 py-1.5 text-sm text-gray-400 hover:text-white bg-white/5 hover:bg-white/10 rounded-lg transition"
            >
              {selectedPhotos.length === mockPhotos.length ? 'Deselect All' : 'Select All'}
            </button>
            <button className="px-3 py-1.5 text-sm font-medium text-black bg-cyan-400 hover:bg-cyan-300 rounded-lg transition flex items-center gap-2">
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              Download {selectedPhotos.length > 0 ? `(${selectedPhotos.length})` : ''}
            </button>
          </div>
        </div>

        {/* Project Header */}
        <div className="px-6 py-4 border-b border-white/10">
          <h1 className="text-xl font-semibold text-white">{selectedProject.name}</h1>
          <p className="text-sm text-gray-500 mt-1">
            {selectedProject.photos} photos • Created {selectedProject.date}
            {selectedProject.status === 'processing' && (
              <span className="ml-2 px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded">Processing</span>
            )}
          </p>
        </div>

        {/* Photo Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4">
            {mockPhotos.map(photo => (
              <div
                key={photo.id}
                onClick={() => togglePhotoSelection(photo.id)}
                className={`aspect-[4/3] rounded-lg overflow-hidden cursor-pointer relative group transition-all ${
                  selectedPhotos.includes(photo.id)
                    ? 'ring-2 ring-cyan-400 ring-offset-2 ring-offset-black'
                    : 'hover:ring-2 hover:ring-white/30 hover:ring-offset-2 hover:ring-offset-black'
                }`}
              >
                <img
                  src={photo.url}
                  alt={photo.name}
                  className="w-full h-full object-cover"
                />
                {/* Checkbox */}
                <div className={`absolute top-2 left-2 w-5 h-5 rounded border-2 flex items-center justify-center transition ${
                  selectedPhotos.includes(photo.id)
                    ? 'bg-cyan-400 border-cyan-400'
                    : 'border-white/50 bg-black/30 opacity-0 group-hover:opacity-100'
                }`}>
                  {selectedPhotos.includes(photo.id) && (
                    <svg className="w-3 h-3 text-black" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
                {/* Hover overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition">
                  <div className="absolute bottom-2 left-2 right-2">
                    <p className="text-xs text-white truncate">{photo.name}</p>
                  </div>
                </div>
              </div>
            ))}
            {/* Add More Photos */}
            <Link
              href="/studio"
              className="aspect-[4/3] rounded-lg border-2 border-dashed border-white/20 hover:border-cyan-400/50 flex flex-col items-center justify-center cursor-pointer transition bg-white/[0.02] hover:bg-white/[0.05]"
            >
              <svg className="w-8 h-8 text-gray-500 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 4v16m8-8H4" />
              </svg>
              <span className="text-sm text-gray-500">Add Photos</span>
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}
