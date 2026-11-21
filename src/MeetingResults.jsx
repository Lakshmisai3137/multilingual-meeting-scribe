import React, { useEffect, useRef, useState } from 'react'
import { Check, Copy, Download, FileAudio2, FileText, Languages, ListChecks, Plus, Search, Sparkles, X } from 'lucide-react'

function stamp(seconds) {
  return String(Math.floor(seconds/60)).padStart(2,'0') + ':' + String(Math.floor(seconds%60)).padStart(2,'0')
}
function subtitleTime(seconds) {
  const ms = Math.max(0, Math.round(seconds*1000))
  return [Math.floor(ms/3600000), Math.floor(ms/60000)%60, Math.floor(ms/1000)%60]
    .map(x => String(x).padStart(2,'0')).join(':') + ',' + String(ms%1000).padStart(3,'0')
}
const headings = { verbatim:'Transcript excerpt', generated:'Meeting summary', no_speech:'No speech detected', unsupported_language:'English insights unavailable' }

export default function MeetingResults({ data, filename, onReset }) {
  const [query,setQuery] = useState('')
  const [checked,setChecked] = useState(new Set())
  const [format,setFormat] = useState('txt')
  const [feedback,setFeedback] = useState('')
  const [manual,setManual] = useState('')
  const timer = useRef()
  const field = useRef()
  useEffect(() => () => clearTimeout(timer.current),[])
  useEffect(() => { if (manual) { field.current?.focus(); field.current?.select() } },[manual])
  const language = data.language === 'und' ? 'Speech language undetermined' :
    'Detected ' + data.language.toUpperCase() + ' (' + (data.language_probability*100).toFixed(1) + '% language confidence) · Transcript ' + data.transcript_language.toUpperCase()
  const segments = data.segments.filter(s => s.text.toLowerCase().includes(query.trim().toLowerCase()))
  const heading = headings[data.summary_kind]
  const tasks = data.items.filter(i => i.kind === 'action').length
  function toggle(i) {
    setChecked(old => { const next = new Set(old); if(next.has(i)) next.delete(i); else next.add(i); return next })
  }
  function notes() {
    return [filename,language,heading,data.summary,'','Actions and decisions',...data.items.map((item,i) =>
      [item.kind === 'action' ? (checked.has(i) ? '[x]' : '[ ]') : 'Decision', item.text,
        item.owner && 'Owner: '+item.owner, item.due && 'Due: '+item.due, stamp(item.start)+'–'+stamp(item.end)].filter(Boolean).join(' · ')),
      '','Transcript', ...data.segments.map(s => stamp(s.start)+'–'+stamp(s.end)+' '+s.text)].join('\n')
  }
  function announce(message) {
    setFeedback(message); clearTimeout(timer.current); timer.current=setTimeout(() => setFeedback(''),2500)
  }
  async function copy() {
    const text = notes()
    try { await navigator.clipboard.writeText(text); announce('Copied meeting notes') } catch { setManual(text) }
  }
  function download() {
    let text
    if (format === 'json') {
      text = JSON.stringify({...data, items:data.items.map((item,i) =>
        item.kind === 'action' ? {...item,completed:checked.has(i)} : item)},null,2)
    } else if (format === 'srt') {
      text = data.segments.map((s,i) => [i+1,subtitleTime(s.start)+' --> '+subtitleTime(s.end),s.text].join('\n')).join('\n\n')+'\n'
    } else text = notes()
    const url=URL.createObjectURL(new Blob([text],{type:format==='json'?'application/json':'text/plain;charset=utf-8'}))
    const link=document.createElement('a')
    link.href=url; link.download=filename.replace(/\.[^.]+$/,'')+' — meeting.'+format
    document.body.appendChild(link); link.click(); link.remove()
    setTimeout(() => URL.revokeObjectURL(url),1000); announce('Exported '+format.toUpperCase())
  }
  return <div className="results-wrap">
    <div className="result-toolbar glass">
      <div className="file-details"><FileAudio2/><div><strong>{filename}</strong><small><Languages size={13}/>{language}</small></div></div>
      <div className="toolbar-actions"><button className="btn" onClick={copy}><Copy/>Copy All</button>
        <select aria-label="Export format" value={format} onChange={e=>setFormat(e.target.value)}>{['txt','srt','json'].map(f=><option key={f}>{f}</option>)}</select>
        <button className="btn" onClick={download}><Download/>Export</button>
        <button className="btn primary" onClick={onReset}><Plus/>New meeting</button></div>
    </div>
    {manual && <div className="manual glass"><label htmlFor="copy-notes">Select and copy your notes</label><textarea ref={field} id="copy-notes" readOnly value={manual}/><button className="btn" onClick={()=>setManual('')}>Done</button></div>}
    <div className="bento"><div className="insight-stack">
      <section className="glass insight"><h2><Sparkles className="cyan"/>{heading}</h2>{data.summary_kind!=='no_speech' && <p>{data.summary}</p>}</section>
      {['verbatim','generated'].includes(data.summary_kind) && <section className="glass insight"><h2><ListChecks className="violet"/>Actions and decisions</h2>
        <p className="completed">{tasks ? checked.size+' of '+tasks+' actions complete' : 'No action items detected'}</p>
        <div className="checklist">{data.items.map((item,i)=>{
          const evidence=<span>{item.text}<small className="evidence">{[item.kind, item.owner && 'Owner: '+item.owner,item.due && 'Due: '+item.due,stamp(item.start)+'–'+stamp(item.end)].filter(Boolean).join(' · ')}</small></span>
          return item.kind==='decision' ? <div className="decision" key={i}>{evidence}</div> :
            <label key={i}><input type="checkbox" checked={checked.has(i)} onChange={()=>toggle(i)}/><span className="check-indicator">{checked.has(i)&&<Check size={12}/>}</span>{evidence}</label>
        })}</div></section>}
    </div><section className="glass insight transcript"><h2><FileText className="cyan"/>Transcript</h2>
      <label className="search"><Search size={17}/><input aria-label="Search transcript" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search transcript…"/>{query&&<button aria-label="Clear search" onClick={()=>setQuery('')}><X size={16}/></button>}</label>
      <div className="segments" tabIndex={0}>{segments.map((s,i)=><article key={i}><time>{stamp(s.start)} – {stamp(s.end)}</time><p>{s.text}</p></article>)}
      {!segments.length&&<p className="empty">{query?'No matching segments.':'No speech segments.'}</p>}</div>
    </section></div><div role="status" className={feedback?'toast':'sr-only'}>{feedback}</div>
  </div>
}
