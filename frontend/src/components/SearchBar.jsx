import { useState } from 'react'

export default function SearchBar({ onSearch, isLoading }) {
  const [query, setQuery] = useState('')
  const [alpha, setAlpha] = useState(0.005)

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      onSearch(query, alpha)
    }
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-4">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Main search input */}
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter search query (e.g., 'Hillary Clinton', 'refugees')"
            className="w-full px-6 py-4 bg-white border border-stone-400 rounded-md text-stone-900 placeholder-stone-500 focus:outline-none focus:border-stone-700 focus:ring-1 focus:ring-stone-700 transition"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading}
            className="absolute right-2 top-1/2 -translate-y-1/2 px-6 py-2 bg-stone-900 text-stone-100 rounded-md hover:bg-stone-700 disabled:opacity-50 disabled:cursor-not-allowed transition font-medium"
          >
            {isLoading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Alpha threshold control */}
        <div className="flex items-center gap-4">
          <label className="text-sm text-stone-700">
            Similarity Threshold (α):
          </label>
          <input
            type="range"
            min="0"
            max="0.1"
            step="0.001"
            value={alpha}
            onChange={(e) => setAlpha(parseFloat(e.target.value))}
            className="flex-1 h-2 bg-stone-300 rounded cursor-pointer"
            disabled={isLoading}
          />
          <span className="text-sm font-mono text-stone-800 min-w-12">
            {alpha.toFixed(3)}
          </span>
        </div>

      </form>
    </div>
  )
}
