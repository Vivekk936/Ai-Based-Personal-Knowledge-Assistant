import { useState, type ChangeEvent } from 'react'
import { ingestFiles, type IngestResult } from '../api'

export function OrganizePanel() {
  const [files, setFiles] = useState<IngestResult[]>([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleUpload(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? [])

    if (selectedFiles.length === 0) return

    setUploading(true)
    setError(null)

    try {
      const result = await ingestFiles(selectedFiles)
      setFiles((prev) => [...result.ingested, ...prev])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setUploading(false)
      event.target.value = ''
    }
  }

  function clearFiles() {
    setFiles([])
    setError(null)
  }

  return (
    <div className="flex h-full flex-col overflow-y-auto p-4">

      <div className="mb-6">
        <h2 className="text-lg font-semibold text-neutral-200">
          Documents
        </h2>

        <p className="mt-1 text-sm text-neutral-500">
          Upload files to make them searchable in Ada.
        </p>
      </div>

      <label
        className={`mb-6 flex w-full cursor-pointer items-center justify-center rounded-lg bg-accent px-4 py-3 text-sm font-medium text-neutral-950 transition hover:bg-accent-strong ${
          uploading ? 'pointer-events-none opacity-50' : ''
        }`}
      >
        {uploading ? 'Uploading...' : '+ Upload files'}

        <input
          type="file"
          multiple
          onChange={handleUpload}
          className="hidden"
          disabled={uploading}
        />
      </label>

      {error && (
        <div className="mb-4 rounded-lg border border-red-900 bg-red-950/30 p-3 text-xs text-red-400">
          {error}
        </div>
      )}

      {files.length === 0 && !uploading && (
        <div className="flex flex-1 items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-neutral-500">
              No files uploaded yet.
            </p>

            <p className="mt-1 text-xs text-neutral-600">
              Click "Upload files" to add documents.
            </p>
          </div>
        </div>
      )}

      {files.length > 0 && (
        <div>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-medium text-neutral-300">
              Uploaded files
            </h3>

            <button
              onClick={clearFiles}
              className="text-xs text-neutral-500 hover:text-red-400"
            >
              Clear
            </button>
          </div>

          <div className="flex flex-col gap-2">
            {files.map((file, index) => (
              <div
                key={`${file.filename}-${index}`}
                className="rounded-lg border border-neutral-800 bg-neutral-900/50 p-3"
              >
                <p className="truncate text-sm text-neutral-200">
                  {file.filename}
                </p>

                <p className="mt-1 text-xs text-neutral-500">
                  {file.state}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

    </div>
  )
}