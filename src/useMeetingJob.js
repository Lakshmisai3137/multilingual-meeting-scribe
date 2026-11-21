import { useEffect, useRef, useState } from 'react'
import { api, submitAudio } from './processMeetingAudio'

const key = 'scribe.job.v1'
function remember(job) {
  try {
    if (job) sessionStorage.setItem(key, JSON.stringify({ id: job.id, filename: job.filename }))
    else sessionStorage.removeItem(key)
  } catch { /* The job still works when browser storage is disabled. */ }
}
function restore() {
  try {
    const job = JSON.parse(sessionStorage.getItem(key))
    return /^[a-f0-9]{32}$/.test(job?.id) ? job : null
  } catch { return null }
}

export default function useMeetingJob() {
  const [job, setJob] = useState(restore)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [sending, setSending] = useState('')
  const busy = useRef(false)
  const id = job?.id
  useEffect(() => {
    if (!id) return
    let timer
    const controller = new AbortController()
    async function poll() {
      const signal = AbortSignal.any([controller.signal, AbortSignal.timeout(15000)])
      try {
        const state = await api('/api/jobs/' + id, { signal })
        if (controller.signal.aborted) return
        setJob(state)
        setError('')
        if (state.state === 'completed') {
          const data = await api('/api/jobs/' + id + '/result', { signal })
          if (!controller.signal.aborted) { setResult(data); window.scrollTo(0,0) }
          return
        }
        if (['failed','cancelled'].includes(state.state)) {
          remember(null); setJob(null); setError(state.error || 'Processing cancelled.')
          return
        }
      } catch (err) {
        if (controller.signal.aborted) return
        if (err.status === 404) { remember(null); setJob(null); setError(err.message); return }
        setError('Connection interrupted. Reconnecting to the local job…')
      }
      timer = setTimeout(poll, 850)
    }
    poll()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [id])

  async function start(file, task) {
    if (busy.current || id) return
    busy.current = true
    setError(''); setSending(file.name)
    try {
      const state = await submitAudio(file, task)
      remember(state); setJob(state)
    } catch (err) {
      setError(err instanceof TypeError ? 'Cannot reach the local engine. Check the backend and API origin.' : err.message)
    } finally { busy.current = false; setSending('') }
  }
  async function cancel() {
    try { setJob(await api('/api/jobs/' + id, { method: 'DELETE', signal: AbortSignal.timeout(15000) })) }
    catch { setError('Could not cancel. Check the connection and try again.') }
  }
  function reset() { remember(null); setResult(null); setJob(null); setError('') }
  return { phase: result ? 'results' : job || sending ? 'processing' : 'idle',
    filename: job?.filename || sending, result, job, error, start, reset, cancel }
}
