export const apiOrigin = (import.meta.env.VITE_API_ORIGIN || 'http://127.0.0.1:8000').replace(/\/$/, '')

export async function api(path, options = {}) {
  const response = await fetch(apiOrigin + path, { cache: 'no-store', ...options })
  const data = await response.json().catch(() => null)
  if (!response.ok || !data) {
    const error = new Error(data?.detail || ('Local engine returned HTTP ' + response.status))
    error.status = response.status
    throw error
  }
  return data
}

export async function submitAudio(file, task) {
  if (!file || !/\.(mp3|wav|m4a)$/i.test(file.name)) throw new Error('Choose an MP3, WAV, or M4A recording.')
  if (!file.size) throw new Error('This file is empty.')
  if (file.size > 128 * 1024 * 1024) throw new Error('Choose a recording under 128 MiB.')
  const body = new FormData()
  body.append('file', file)
  body.append('task', task)
  return api('/api/jobs', { method: 'POST', body, signal: AbortSignal.timeout(135000) })
}
