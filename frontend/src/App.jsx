import { useState } from 'react'
import axios from 'axios'
import SearchBar from '@/components/SearchBar'
import ResultsList from '@/components/ResultsList'

// Always use relative API paths.
// In dev, Vite proxies /api to the FastAPI server running on port 8001.
const API_BASE = '/api'

export default function App() {
  const [results, setResults] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [searchMetrics, setSearchMetrics] = useState(null)
  const [error, setError] = useState(null)
  const [showResults, setShowResults] = useState(false)
  const [speechByDoc, setSpeechByDoc] = useState({})
  const [expandedDocId, setExpandedDocId] = useState(null)
  const [loadingSpeechId, setLoadingSpeechId] = useState(null)

  const handleSearch = async (query, alpha) => {
    setIsLoading(true)
    setError(null)
    setShowResults(true)
    setSpeechByDoc({})
    setSearchMetrics(null)
    setExpandedDocId(null)
    setLoadingSpeechId(null)

    try {
      const response = await axios.post(`${API_BASE}/search`, {
        query,
        alpha,
      })

      setResults(response.data.results)
      setSearchMetrics(response.data.metrics || null)
    } catch (err) {
      console.error('Search failed:', err)
      setError(err.response?.data?.detail || 'Search failed. Please try again.')
      setResults([])
      setSearchMetrics(null)
    } finally {
      setIsLoading(false)
    }
  }

  const handleToggleExpand = async (docId) => {
    if (expandedDocId === docId) {
      setExpandedDocId(null)
      return
    }

    setExpandedDocId(docId)
    if (speechByDoc[docId]) {
      return
    }

    setLoadingSpeechId(docId)
    try {
      const response = await axios.get(`${API_BASE}/speeches/${docId}`)
      setSpeechByDoc((prev) => ({
        ...prev,
        [docId]: response.data.text || '',
      }))
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load speech content.')
      setSpeechByDoc((prev) => ({
        ...prev,
        [docId]: '',
      }))
    } finally {
      setLoadingSpeechId(null)
    }
  }

  const retrievedDocIds = results.map((result) => result.doc_id).join(', ')

  return (
    <div className="min-h-screen bg-stone-100 text-stone-900">
      <div className="bg-stone-50 border-b border-stone-300 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-3 md:py-4 space-y-4">
          <div className="space-y-1">
            <p className="tracking-[0.25em] uppercase text-[10px] text-stone-500">IR News Desk</p>
            <h1 className="text-3xl md:text-4xl font-bold newspaper-title">Vector Space Model Retrieval</h1>
            <p className="text-sm md:text-base text-stone-700 max-w-3xl">
              Explore ranked retrieval over a real speech corpus using TF-IDF weighting and cosine similarity scoring.
            </p>
          </div>
          <SearchBar onSearch={handleSearch} isLoading={isLoading} />
        </div>
      </div>

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-6">
        {error && (
          <div className="news-alert">
            {error}
          </div>
        )}

        <section className="space-y-6">
          {showResults ? (
            <div className="news-card p-5 md:p-6 w-full">
              <h2 className="text-2xl font-semibold mb-4 newspaper-subtitle">Ranked Results</h2>
              <div className="mb-3 flex flex-wrap gap-2 text-sm">
                <span className="query-chip">Precision: {(searchMetrics?.precision ?? 0).toFixed(3)}</span>
                <span className="query-chip">Recall: {(searchMetrics?.recall ?? 0).toFixed(3)}</span>
                <span className="query-chip">Accuracy: {(searchMetrics?.accuracy ?? 0).toFixed(3)}</span>
                <span className="query-chip">F1: {(searchMetrics?.f1 ?? 0).toFixed(3)}</span>
              </div>
              {results.length > 0 && (
                <div className="mb-3 text-sm text-stone-600">
                  Retrieved document IDs: <span className="font-semibold text-stone-800">{retrievedDocIds}</span>
                </div>
              )}
              <ResultsList
                results={results}
                isLoading={isLoading}
                expandedDocId={expandedDocId}
                onToggleExpand={handleToggleExpand}
                speechByDoc={speechByDoc}
                loadingSpeechId={loadingSpeechId}
              />
            </div>
          ) : (
            <div className="hidden" />
          )}
        </section>
      </main>
    </div>
  )
}
