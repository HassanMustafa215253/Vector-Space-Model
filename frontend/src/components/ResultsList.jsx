export default function ResultsList({
  results,
  isLoading,
  expandedDocId,
  onToggleExpand,
  speechByDoc,
  loadingSpeechId,
}) {
  if (isLoading) {
    return (
      <div className="space-y-3">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-20 bg-stone-200 rounded animate-pulse" />
        ))}
      </div>
    )
  }

  if (results.length === 0) {
    return (
      <div className="text-center py-12 text-stone-500">
        <p>No results yet. Try searching for something!</p>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {results.map((result, idx) => (
        <button
          key={result.doc_id}
          type="button"
          onClick={() => onToggleExpand(result.doc_id)}
          className="w-full text-left news-card p-4 hover:bg-stone-100 transition cursor-pointer group"
        >
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-mono text-stone-600">
                  #{idx + 1}
                </span>
                <span className="text-lg font-semibold text-stone-900">
                  Speech {result.doc_id}
                </span>
                <span className="text-xs text-stone-500">(click to expand)</span>
              </div>
              <p className="text-sm text-stone-700 line-clamp-2">
                {result.title || 'No preview available'}
              </p>
            </div>
            <div className="ml-4 text-right">
              <div className="text-2xl font-bold text-stone-900">
                {(result.score * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-stone-500">relevance</div>
            </div>
          </div>
          {/* Score bar */}
          <div className="mt-2 h-1.5 bg-stone-200 rounded overflow-hidden">
            <div
              className="h-full bg-stone-700 transition-all duration-500"
              style={{ width: `${Math.min(result.score * 100, 100)}%` }}
            />
          </div>

          {expandedDocId === result.doc_id && (
            <div className="mt-3 rounded-md border border-stone-300 bg-stone-50 p-4">
              {loadingSpeechId === result.doc_id ? (
                <p className="text-sm text-stone-600">Loading full speech...</p>
              ) : speechByDoc[result.doc_id] ? (
                <p className="text-sm leading-7 text-stone-800 whitespace-pre-wrap">
                  {speechByDoc[result.doc_id]}
                </p>
              ) : (
                <p className="text-sm text-red-700">Full speech is unavailable.</p>
              )}
            </div>
          )}
        </button>
      ))}
    </div>
  )
}
