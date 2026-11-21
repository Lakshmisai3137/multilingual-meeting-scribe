import React, { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { AudioLines, Check, CloudUpload, FileAudio2, Fingerprint, LoaderCircle, Waves } from 'lucide-react'
import NeuralScene, { OrbSlot } from './NeuralScene'
import MeetingResults from './MeetingResults'
import { apiOrigin } from './processMeetingAudio'
import useMeetingJob from './useMeetingJob'

function Brand() {
  return <div className="brand flex items-center"><span className="brand-glyph" aria-hidden="true">{[9,19,29,18,10].map((h,i) => <i key={i} style={{height:h}}/>)}</span><span>SCRIBE<span className="brand-ai">.AI</span></span></div>
}

function EngineStatus({ busy }) {
  const [status, setStatus] = useState('Checking')
  useEffect(() => {
    const controller = new AbortController()
    let timer
    async function check() {
      try {
        const response = await fetch(apiOrigin + '/api/health', {
          cache: 'no-store', signal: AbortSignal.any([controller.signal, AbortSignal.timeout(8000)]),
        })
        const health = response.ok && await response.json()
        if (!controller.signal.aborted) setStatus(health.status === 'ok' ? 'Ready' : 'Offline')
      } catch {
        if (!controller.signal.aborted) setStatus('Offline')
      } finally {
        if (!controller.signal.aborted) timer = setTimeout(check, 5000)
      }
    }
    check()
    return () => { controller.abort(); clearTimeout(timer) }
  }, [])
  return <div className="engine-pill" role="status">
    <i style={status === 'Ready' ? undefined : {background: 'currentColor', boxShadow: 'none', color: '#89909a'}}/>
    Neural Engine {status === 'Ready' && busy ? 'Active' : status}
  </div>
}

function ProgressRail({ phase }) {
  const index = { idle:0, processing:1, results:2 }[phase]
  return <ol aria-label="Meeting workflow" className="rail">{['Upload','Process','Insights'].map((label,i) =>
    <li key={label} aria-current={i === index ? 'step' : undefined} className={i <= index ? 'active' : ''}>
      <span>{i < index ? <Check size={10}/> : '0' + (i+1)}</span>{label}</li>)}</ol>
}

function SignalWing({ right = false, busy }) {
  return <div className={'signal-wing ' + (right ? 'right' : 'left')} aria-hidden="true">
    <div className="signal-caption"><span className="signal-mark"/> {right ? 'ORB STATE' : 'WORKSPACE'}</div>
    <div className="signal-trace">{Array.from({length:74},(_,i) => {
      const envelope = Math.sin((i+2)/76*Math.PI)**1.3
      const height = 2+envelope*(7+Math.abs(Math.sin(i*1.85)*Math.sin(i*.24+.8))*27)
      return <i key={i} style={{height,animationDelay:(i%9)*-.13+'s'}}/>
    })}</div>
    <div className="signal-readout">{right ? (busy ? 'PROCESSING' : 'STANDBY') : (busy ? 'PROCESSING' : 'AWAITING INPUT')}</div>
  </div>
}

function Processing({ filename, job, onCancel, error }) {
  return <div className="processing-content" role="status">
    <div className="processing-file"><LoaderCircle size={18} className="spin"/><strong>{filename}</strong></div>
    <p>{({queued:'Waiting for the engine',decoding:'Decoding audio',transcribing:'Transcribing',translating:'Translating to English',summarizing:'Summarizing passages',synthesizing:'Synthesizing meeting summary',complete:'Loading results'})[job?.stage] || (job ? 'Restoring job' : 'Uploading recording')}</p>
    {job?.total > 0 && <><progress max={job.total} value={job.current} aria-label={job.stage}/><small>{['transcribing','translating'].includes(job.stage) ? Math.floor(job.current)+' / '+Math.ceil(job.total)+' audio seconds' : job.current+' / '+job.total+' completed'}</small></>}
    {job && <button className="btn" disabled={job.state === 'cancelling'} onClick={onCancel}>{job.state === 'cancelling' ? 'Cancelling…' : 'Cancel job'}</button>}
    {error && <p role="alert">{error}</p>}
  </div>
}

function Intake({ phase, filename, onProcess, error, task, onTask, job, onCancel }) {
  const picker = useRef()
  const dragDepth = useRef(0)
  const [dragging,setDragging] = useState(false)
  const busy = phase === 'processing'
  return <motion.section layout transition={{layout:{duration:.32,ease:[.22,1,.36,1]}}} className={'intake glass ' + (dragging ? 'dragging' : '')}>
    <div className="intake-head"><div className="flex items-center gap-2.5"><AudioLines size={17}/><h2>{busy ? 'Processing audio' : 'Audio input'}</h2></div>
      <span className="mode"><i/>{busy ? 'PROCESSING' : 'READY TO UPLOAD'}</span></div>
    <AnimatePresence mode="wait" initial={false}>
      {busy ? <motion.div key="process" {...panelMotion}><Processing filename={filename} job={job} onCancel={onCancel} error={error}/></motion.div> :
      <motion.div key="upload" {...panelMotion}>
        <label className="task-select">Transcript language <select value={task} onChange={e=>onTask(e.target.value)}>
          <option value="transcribe">Transcribe in original language</option><option value="translate">Translate to English</option></select></label>
        <p className="task-help">{task === 'transcribe' ? 'English transcripts include summaries and action items.' : 'Generate an English transcript, summary, and action items.'}</p>
        <input className="sr-only" type="file" tabIndex={-1} ref={picker} accept=".mp3,.wav,.m4a" aria-label="Choose meeting audio"
          onChange={e => { const file=e.target.files[0]; e.target.value=''; if(file) onProcess(file) }}/>
        <button className="dropzone" onClick={() => picker.current?.click()}
          onDragEnter={e => {e.preventDefault(); dragDepth.current++; setDragging(true)}}
          onDragOver={e => {e.preventDefault(); e.dataTransfer.dropEffect='copy'}}
          onDragLeave={e => {e.preventDefault(); if (--dragDepth.current<=0) setDragging(false)}}
          onDrop={e => {e.preventDefault(); dragDepth.current=0; setDragging(false); const file=e.dataTransfer.files[0]; if(file) onProcess(file)}}>
          <span className="upload-tile"><CloudUpload size={28} strokeWidth={1.5}/></span>
          <span className="drop-copy"><strong>{dragging ? 'Release to process audio' : 'Drop your meeting audio here'}</strong><small>or <span>browse files</span> on your device</small></span>
          <span className="formats">{['MP3','WAV','M4A'].map(x => <span key={x}>{x}</span>)}</span>
        </button>
        {error && <p className="error" role="alert">{error}</p>}
      </motion.div>}
    </AnimatePresence>
  </motion.section>
}

const panelMotion = {
  initial: {opacity:0, y:8},
  animate: {opacity:1, y:0},
  exit: {opacity:0, y:-4, transition:{duration:.13}},
  transition: {duration:.34, ease:[.22,1,.36,1]},
}

export default function App() {
  const { phase, result, filename, job, error, start, cancel, reset: clearJob } = useMeetingJob()
  const [task,setTask] = useState('transcribe')
  const [ready,setReady] = useState(false)
  const anchor = useRef()
  const heading = useRef()
  const resultView = phase === 'results'
  useEffect(() => { if (phase === 'results') heading.current?.focus({preventScroll:true}) },[phase])
  function reset() {
    clearJob()
    window.scrollTo(0,0)
    requestAnimationFrame(() => heading.current?.focus({preventScroll:true}))
  }
  return <div className={'app phase-' + phase}>
    <div className="space-bg" aria-hidden="true"/>
    <NeuralScene phase={phase} anchor={anchor} onReady={setReady}/>
    <header className="navbar"><div className="nav-inner"><Brand/><EngineStatus busy={phase === 'processing'}/></div></header>
    <main className={resultView ? 'main-content results' : 'main-content'}>
      <div className="hero" key={resultView ? 'insights-title' : 'upload-title'}><ProgressRail phase={phase}/>
        <h1 ref={heading} tabIndex={-1}>{resultView ? <>Meeting <em>insights</em></> : <>Multilingual <em>Meeting Scribe</em></>}</h1>
        <p>{resultView ? filename : 'Turn meeting audio into transcripts, summaries, and clear action items.'}</p>
      </div>
      {!resultView ? <>
        <div className="orb-view"><SignalWing busy={phase === 'processing'}/><OrbSlot anchor={anchor} ready={ready}/><SignalWing right busy={phase === 'processing'}/></div>
        <Intake phase={phase} filename={filename} onProcess={file => start(file,task)} error={error} task={task} onTask={setTask} job={job} onCancel={cancel}/>
      </> : <motion.div initial={{opacity:0,y:12}} animate={{opacity:1,y:0}} transition={{duration:.45}}>
        <MeetingResults data={result} filename={filename} onReset={reset}/>
      </motion.div>}
    </main>
    <footer><div className="footer-inner"><span><Fingerprint size={15}/>Local workspace</span></div></footer>
  </div>
}
