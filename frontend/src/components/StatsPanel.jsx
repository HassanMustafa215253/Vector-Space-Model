import { useState } from 'react'

export default function StatsPanel({ stats }) {
  const [expanded, setExpanded] = useState(true)

  if (!stats) return null

  return (
    <div className="news-card p-5">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between hover:text-stone-600 transition"
      >
        <h3 className="font-semibold text-xl newspaper-subtitle">VSM Statistics</h3>
        <svg
          className={`w-5 h-5 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 14l-7 7m0 0l-7-7m7 7V3"
          />
        </svg>
      </button>

      {expanded && (
        <div className="mt-4 space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-stone-600">Total Documents:</span>
            <span className="text-stone-900 font-semibold">
              {stats.total_documents}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-stone-600">Vocabulary Size:</span>
            <span className="text-stone-900 font-semibold">
              {stats.vocabulary_size?.toLocaleString()}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-stone-600">Avg Document Frequency:</span>
            <span className="text-stone-900 font-semibold">
              {(stats.avg_df || 0).toFixed(2)}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
