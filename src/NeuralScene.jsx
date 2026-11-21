import React, { Component, lazy, Suspense, useEffect, useMemo, useState } from 'react'
import { surface } from './orbGeometry'

const WebGLScene = lazy(() => import('./WebGLScene'))

class SceneBoundary extends Component {
  state = { failed: false }
  static getDerivedStateFromError() { return { failed: true } }
  componentDidCatch() { this.props.onFailure() }
  render() { return this.state.failed ? null : this.props.children }
}

function WireFallback() {
  const paths = useMemo(() => Array.from({ length: 65 }, (_, row) => {
    const points = Array.from({ length: 131 }, (_, col) => {
      const [x, y, z] = surface((row / 64 - .5) * Math.PI, col / 130 * Math.PI)
      const y2 = y * .95 - z * .31
      return [(x * .94 - y2 * .34) * 89 + 110, 111 - (x * .34 + y2 * .94) * 89].map(v => v.toFixed(2)).join(',')
    })
    return 'M' + points.join(' L')
  }), [])
  return <svg className="fallback-orb" viewBox="-55 -14 330 250" aria-hidden="true">
    <defs><linearGradient id="wire-color"><stop stopColor="#28f3f0"/><stop offset=".47" stopColor="#59a8fc"/><stop offset=".78" stopColor="#8460fc"/><stop offset="1" stopColor="#df91ff"/></linearGradient></defs>
    <g fill="none" stroke="url(#wire-color)">{paths.map((d,i) => <path key={i} d={d} strokeWidth=".85"/>)}
    <ellipse cx="110" cy="108" rx="142" ry="38" transform="rotate(-16 110 108)" opacity=".6"/>
    <ellipse cx="110" cy="108" rx="135" ry="47" transform="rotate(24 110 108)" opacity=".3"/></g>
  </svg>
}

export function OrbSlot({ anchor, ready }) {
  return <div ref={anchor} className="orb-slot" aria-hidden="true">
    <div className="orb-aura"/>{!ready && <WireFallback/>}
  </div>
}

export default function NeuralScene({ anchor, phase, onReady }) {
  const [visible,setVisible] = useState(!document.hidden)
  useEffect(() => {
    const change = () => setVisible(!document.hidden)
    document.addEventListener('visibilitychange',change)
    return () => document.removeEventListener('visibilitychange', change)
  }, [])
  return <div className={'scene-layer ' + (phase === 'results' ? 'scene-dim' : '')} aria-hidden="true">
    <SceneBoundary onFailure={() => onReady(false)}><Suspense fallback={null}>
      <WebGLScene anchor={anchor} phase={phase} visible={visible} onReady={onReady}/>
    </Suspense></SceneBoundary>
  </div>
}
