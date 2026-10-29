import { useEffect, useState, type ChangeEvent } from 'react'
import { getEvalRag, uploadEvalPdf, type RagEvalReport } from '../api'

export function EvalPanel() {
  const [report, setReport] = useState<RagEvalReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)

  useEffect(() => {
    loadReport(false)
  }, [])

  async function loadReport(forceRefresh = false) {
    setLoading(true)
    setError(null)
    try {
      const data = await getEvalRag(forceRefresh)
      setReport(data)
      if (data.individual_cases && data.individual_cases.length > 0) {
        setSelectedCaseId(data.individual_cases[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load evaluation benchmark')
    } finally {
      setLoading(false)
    }
  }

  async function handlePdfUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    setUploading(true)
    setError(null)
    try {
      const data = await uploadEvalPdf(file)
      setReport(data)
      if (data.individual_cases && data.individual_cases.length > 0) {
        setSelectedCaseId(data.individual_cases[0].id)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'PDF evaluation upload failed')
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  const selectedCase = report?.individual_cases?.find((c) => c.id === selectedCaseId)

  return (
    <div className="flex h-full flex-col overflow-y-auto p-6">
      {/* Header */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-neutral-800/80 pb-5">
        <div>
          <div className="text-[11px] font-bold tracking-wider text-accent uppercase">Quality Control Benchmark</div>
          <h2 className="text-xl font-semibold text-neutral-100">RAG Evaluation Benchmark</h2>
          <p className="mt-1 text-xs text-neutral-500">
            Empirical measurement of retrieval accuracy, answer faithfulness, and latency on real uploaded documents.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => loadReport(true)}
            disabled={loading || uploading}
            className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-neutral-950 transition hover:bg-accent-strong disabled:opacity-50"
          >
            {loading ? 'Evaluating...' : 'Execute Benchmark'}
          </button>
          <button
            onClick={() => loadReport(false)}
            disabled={loading || uploading}
            className="rounded-lg border border-neutral-800 bg-neutral-900 px-3 py-1.5 text-xs font-medium text-neutral-300 transition hover:border-neutral-700 hover:text-neutral-100 disabled:opacity-50"
          >
            Reload Results
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-lg border border-red-900/60 bg-red-950/30 p-3 text-xs text-red-400">
          {error}
        </div>
      )}

      {/* Meta status bar */}
      {report?.benchmark_summary && (
        <div className="mb-5 flex flex-wrap items-center gap-3 rounded-lg border border-neutral-800/70 bg-neutral-900/40 px-3 py-2 text-xs text-neutral-400">
          <span>
            Document Under Test: <strong className="text-neutral-200">{report.benchmark_summary.source_pdf || 'None'}</strong>
          </span>
          <span className="text-neutral-600">·</span>
          <span>
            Mode: <strong className="text-neutral-200">{report.benchmark_summary.mode?.toUpperCase() || 'ONLINE'}</strong>
          </span>
          <span className="text-neutral-600">·</span>
          <span>Timestamp: {report.benchmark_summary.timestamp || 'Recent'}</span>
        </div>
      )}

      {/* Empty State: No Uploaded PDF */}
      {(!report || report.status === 'no_uploaded_pdfs') && !loading && (
        <div className="my-8 flex flex-col items-center justify-center rounded-xl border border-dashed border-neutral-800 p-10 text-center">
          <h3 className="text-sm font-semibold text-neutral-200">No Document Detected for Evaluation</h3>
          <p className="mt-1.5 max-w-md text-xs text-neutral-500">
            Ada generates evaluation questions dynamically from <strong>real document text</strong> without using dummy mock data.
            Upload a PDF below to benchmark retrieval accuracy, correctness, faithfulness, and latency.
          </p>

          <label className="mt-5 inline-flex cursor-pointer items-center rounded-lg bg-accent px-4 py-2 text-xs font-semibold text-neutral-950 transition hover:bg-accent-strong">
            {uploading ? 'Ingesting and benchmarking...' : 'Upload PDF to Benchmark'}
            <input
              type="file"
              accept=".pdf"
              onChange={handlePdfUpload}
              className="hidden"
              disabled={uploading}
            />
          </label>
        </div>
      )}

      {/* 4 Core Pillars KPI Cards */}
      {report?.status === 'success' && (
        <>
          <div className="mb-6 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {/* 1. Retrieval Accuracy */}
            <div className="rounded-xl border border-neutral-800/80 bg-neutral-900/60 p-4">
              <div className="text-[11px] font-bold tracking-wider text-neutral-500 uppercase">1. Retrieval Accuracy</div>
              <div className="mt-2 text-2xl font-bold text-neutral-100">
                {((report.retrieval_accuracy?.hit_rate_at_3 ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-xs text-neutral-400">
                MRR: {(report.retrieval_accuracy?.mean_mrr ?? 0).toFixed(2)} · Hit@1: {((report.retrieval_accuracy?.hit_rate_at_1 ?? 0) * 100).toFixed(0)}%
              </div>
            </div>

            {/* 2. Answer Correctness */}
            <div className="rounded-xl border border-neutral-800/80 bg-neutral-900/60 p-4">
              <div className="text-[11px] font-bold tracking-wider text-neutral-500 uppercase">2. Answer Correctness</div>
              <div className="mt-2 text-2xl font-bold text-neutral-100">
                {((report.correctness?.overall_correctness ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-xs text-neutral-400">
                Recall: {((report.correctness?.mean_fact_recall ?? 0) * 100).toFixed(0)}% · F1: {((report.correctness?.mean_token_f1 ?? 0) * 100).toFixed(0)}%
              </div>
            </div>

            {/* 3. Faithfulness */}
            <div className="rounded-xl border border-neutral-800/80 bg-neutral-900/60 p-4">
              <div className="text-[11px] font-bold tracking-wider text-neutral-500 uppercase">3. Grounded Faithfulness</div>
              <div className="mt-2 text-2xl font-bold text-neutral-100">
                {((report.faithfulness?.mean_faithfulness ?? 0) * 100).toFixed(1)}%
              </div>
              <div className="mt-1 text-xs text-neutral-400">
                Hallucination: {((report.faithfulness?.mean_hallucination_rate ?? 0) * 100).toFixed(0)}% · Bad Nums: {report.faithfulness?.total_unsupported_numbers ?? 0}
              </div>
            </div>

            {/* 4. Response Time */}
            <div className="rounded-xl border border-neutral-800/80 bg-neutral-900/60 p-4">
              <div className="text-[11px] font-bold tracking-wider text-neutral-500 uppercase">4. Response Latency</div>
              <div className="mt-2 text-2xl font-bold text-neutral-100">
                {(report.response_time_ms?.total?.mean_ms ?? 0).toFixed(1)} ms
              </div>
              <div className="mt-1 text-xs text-neutral-400">
                P50: {(report.response_time_ms?.total?.median_ms ?? 0).toFixed(1)}ms · Ret: {(report.response_time_ms?.retrieval?.mean_ms ?? 0).toFixed(1)}ms
              </div>
            </div>
          </div>

          {/* Dynamic Q&A Cases Table */}
          <div className="mb-6 rounded-xl border border-neutral-800/80 bg-neutral-900/40 p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-xs font-bold tracking-wider text-neutral-400 uppercase">
                Dynamic PDF Benchmark Cases ({report.individual_cases?.length ?? 0})
              </h3>
              <label className="cursor-pointer text-xs font-medium text-accent hover:underline">
                {uploading ? 'Uploading...' : '+ Benchmark Another PDF'}
                <input
                  type="file"
                  accept=".pdf"
                  onChange={handlePdfUpload}
                  className="hidden"
                  disabled={uploading}
                />
              </label>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead>
                  <tr className="border-b border-neutral-800 text-[11px] font-semibold text-neutral-500 uppercase">
                    <th className="py-2 pr-3">ID</th>
                    <th className="py-2 pr-3">Category</th>
                    <th className="py-2 pr-3">Question Extracted from PDF</th>
                    <th className="py-2 pr-3">Retrieval</th>
                    <th className="py-2 pr-3">MRR</th>
                    <th className="py-2 pr-3">Correctness</th>
                    <th className="py-2 pr-3">Faithfulness</th>
                    <th className="py-2">Latency</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/60">
                  {report.individual_cases?.map((c) => {
                    const isHit = (c.retrieval?.hit_at_3 ?? 0) > 0.5
                    const isOut = c.retrieval?.is_out_of_corpus
                    const selected = c.id === selectedCaseId

                    return (
                      <tr
                        key={c.id}
                        onClick={() => setSelectedCaseId(c.id)}
                        className={`cursor-pointer transition hover:bg-neutral-800/40 ${selected ? 'bg-neutral-800/60' : ''}`}
                      >
                        <td className="py-2.5 pr-3 font-mono text-neutral-400">{c.id}</td>
                        <td className="py-2.5 pr-3 font-medium text-neutral-300">{c.category.replace(/_/g, ' ').toUpperCase()}</td>
                        <td className="py-2.5 pr-3 text-neutral-200">{c.query}</td>
                        <td className="py-2.5 pr-3">
                          {isOut ? (
                            <span className="rounded bg-neutral-800 px-1.5 py-0.5 text-[10px] text-neutral-400">N/A</span>
                          ) : isHit ? (
                            <span className="rounded bg-emerald-950/60 px-1.5 py-0.5 text-[10px] font-semibold text-emerald-400 border border-emerald-800/50">HIT</span>
                          ) : (
                            <span className="rounded bg-red-950/60 px-1.5 py-0.5 text-[10px] font-semibold text-red-400 border border-red-800/50">MISS</span>
                          )}
                        </td>
                        <td className="py-2.5 pr-3 text-neutral-400">{isOut ? '--' : (c.retrieval?.mrr ?? 0).toFixed(2)}</td>
                        <td className="py-2.5 pr-3 text-neutral-300">{((c.correctness?.correctness ?? 0) * 100).toFixed(1)}%</td>
                        <td className="py-2.5 pr-3 text-neutral-300">{((c.faithfulness?.faithfulness ?? 0) * 100).toFixed(1)}%</td>
                        <td className="py-2.5 font-mono text-neutral-400">{(c.timing_ms?.total ?? 0).toFixed(1)} ms</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Deep-Dive Case Inspector */}
          {selectedCase && (
            <div className="rounded-xl border border-neutral-800/80 bg-neutral-900/60 p-4">
              <h3 className="mb-3 text-xs font-bold tracking-wider text-neutral-400 uppercase">
                Deep-Dive Inspection: Query {selectedCase.id}
              </h3>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 text-xs">
                <div className="space-y-2 rounded-lg bg-neutral-950/50 p-3">
                  <div>
                    <span className="font-semibold text-neutral-400">Query: </span>
                    <span className="text-neutral-200">{selectedCase.query}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-neutral-400">Reference Ground Truth: </span>
                    <span className="text-neutral-300">{selectedCase.reference_answer || 'N/A'}</span>
                  </div>
                </div>

                <div className="space-y-2 rounded-lg bg-neutral-950/50 p-3">
                  <div>
                    <span className="font-semibold text-neutral-400">Generated Answer: </span>
                    <span className="text-neutral-200">{selectedCase.generated_answer || 'None'}</span>
                  </div>
                  <div>
                    <span className="font-semibold text-neutral-400">Citations: </span>
                    <span className="font-mono text-accent">{selectedCase.citations?.join(', ') || 'None'}</span>
                  </div>
                  <div className="text-[11px] text-neutral-500">
                    Latency: Total {(selectedCase.timing_ms?.total ?? 0).toFixed(1)}ms (Ret: {(selectedCase.timing_ms?.retrieval ?? 0).toFixed(1)}ms, Gen: {(selectedCase.timing_ms?.generation ?? 0).toFixed(1)}ms)
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
