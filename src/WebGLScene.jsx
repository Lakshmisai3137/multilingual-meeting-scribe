import React, { useEffect, useLayoutEffect, useMemo, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Float } from '@react-three/drei'
import { useReducedMotion } from 'framer-motion'
import * as THREE from 'three'
import { makeFilaments } from './orbGeometry'

const vertex = `
  uniform float time;
  uniform float energy;
  varying vec3 point;
  varying float depth;
  void main(){
    vec3 p = position;
    p *= 1.0 + (.013 + .022 * energy) * sin(p.y*7.0 + time*1.7 + p.x*3.0);
    vec4 viewPoint = modelViewMatrix * vec4(p, 1.0);
    point = p;
    depth = (modelViewMatrix * vec4(p, 0.0)).z;
    gl_Position = projectionMatrix * viewPoint;
  }`
const fragment = `
  uniform float time;
  uniform float energy;
  varying vec3 point;
  varying float depth;
  void main(){
    float band = smoothstep(-.66,.72, point.x + .20*sin(point.y*4.0+time*.24) - energy*.24);
    vec3 color = mix(vec3(.06,.93,.99), vec3(.68,.34,1.), band);
    float striation = .5 + .5*sin(17.0*point.y + 6.0*point.x + 6.0*point.z + time*.3);
    float shine = pow(striation,4.0);
    color = mix(color, vec3(.83,.95,1.), shine*.37);
    float bright = .20 + .38*striation + .40*shine;
    gl_FragColor = vec4(color, bright * smoothstep(-.8,.36,depth));
  }`

function Filaments({ uniforms }) {
  const positions = useMemo(makeFilaments, [])
  return <lineSegments>
    <bufferGeometry><bufferAttribute attach="attributes-position" args={[positions,3]}/></bufferGeometry>
    <shaderMaterial vertexShader={vertex} fragmentShader={fragment} uniforms={uniforms}
      transparent depthWrite={false} blending={THREE.AdditiveBlending}/>
  </lineSegments>
}

function Orbit({ rotation, radius, color, opacity, dot }) {
  return <group rotation={rotation}>
    <mesh><torusGeometry args={[radius,.0023,4,168]}/><meshBasicMaterial color={color} transparent opacity={opacity} depthWrite={false}/></mesh>
    <mesh position={[Math.cos(dot)*radius,Math.sin(dot)*radius,0]}>
      <sphereGeometry args={[.016,10,8]}/><meshBasicMaterial color={color}/>
    </mesh>
  </group>
}

function Orb({ anchor, phase, reduced, onReady }) {
  const { viewport, size, camera, gl, invalidate } = useThree()
  const root = useRef()
  const shape = useRef()
  const pointer = useRef({ x: 0, y: 0 })
  const target = useRef({ x: 0, y: 0, scale: 1 })
  const placed = useRef(false)
  const uniforms = useMemo(() => ({time:{value:.5},energy:{value:0}}),[])
  useLayoutEffect(() => {
    function measure() {
      const box = anchor.current?.getBoundingClientRect()
      const behind = phase === 'results' || !box
      const x = behind ? size.width * .86 : box.left + box.width/2
      const y = behind ? size.height * .26 : box.top + box.height/2
      const units = viewport.width / size.width
      target.current = {
        x: (x - size.width/2)*units,
        y: (size.height/2 - y)*units,
        scale: (behind ? 90 : Math.min(box.height*.41,box.width*.30)) * units,
      }
      if (!placed.current && root.current) {
        root.current.position.set(target.current.x,target.current.y,0)
        root.current.scale.setScalar(target.current.scale)
        placed.current = true
      }
      invalidate()
    }
    measure()
    const observer = new ResizeObserver(measure)
    if (anchor.current) observer.observe(anchor.current)
    observer.observe(document.body)
    window.addEventListener('scroll',measure,{passive:true})
    return () => { observer.disconnect(); window.removeEventListener('scroll',measure) }
  },[anchor,phase,size,viewport.width,invalidate])
  useEffect(() => {
    function move(e) { pointer.current.x=e.clientX/window.innerWidth*2-1; pointer.current.y=1-e.clientY/window.innerHeight*2 }
    const lost = e => { e.preventDefault(); onReady(false) }
    const restored = () => { onReady(true); invalidate() }
    window.addEventListener('pointermove',move,{passive:true})
    gl.domElement.addEventListener('webglcontextlost',lost)
    gl.domElement.addEventListener('webglcontextrestored',restored)
    onReady(true)
    invalidate()
    return () => {
      window.removeEventListener('pointermove',move)
      gl.domElement.removeEventListener('webglcontextlost',lost)
      gl.domElement.removeEventListener('webglcontextrestored',restored)
    }
  },[gl,invalidate,onReady])
  useFrame((_,elapsed) => {
    const dt = Math.min(elapsed,.05)
    const k = reduced ? 1 : 1-Math.exp(-dt*5)
    const energy = uniforms.energy
    energy.value += ((phase === 'processing' ? 1 : 0)-energy.value)*k
    uniforms.time.value += reduced ? 0 : dt*(.48+energy.value*3)
    const t = uniforms.time.value
    const {x,y,scale} = target.current
    root.current.position.x += (x-root.current.position.x)*k
    root.current.position.y += (y-root.current.position.y)*k
    root.current.scale.setScalar(root.current.scale.x+(scale-root.current.scale.x)*k)
    shape.current.rotation.set(.22+Math.sin(t*.25)*.12,-.34+t*.09,-.34)
    if (!reduced) {
      root.current.rotation.y += (pointer.current.x*.20-root.current.rotation.y)*k
      root.current.rotation.x += (-pointer.current.y*.12-root.current.rotation.x)*k
      camera.position.x += (pointer.current.x*.055-camera.position.x)*k
      camera.position.y += (pointer.current.y*.04-camera.position.y)*k
    }
  })
  return <group ref={root}>
    <Float speed={reduced ? 0 : 1.4} rotationIntensity={.10} floatIntensity={.14} floatingRange={[-.13,.13]}>
      <group ref={shape}>
        <mesh><sphereGeometry args={[.86,48,32]}/><meshBasicMaterial color="#081123"/></mesh>
        <Filaments uniforms={uniforms}/>
      </group>
      <Orbit radius={1.39} rotation={[1.29,.24,-.28]} color="#69dae6" opacity={.50} dot={2.7}/>
      <Orbit radius={1.28} rotation={[1.06,-.51,.32]} color="#9a87ed" opacity={.40} dot={.68}/>
    </Float>
  </group>
}

export default function WebGLScene({ anchor, phase, visible, onReady }) {
  const reduced = useReducedMotion()
  return <Canvas orthographic camera={{ position:[0,0,10], zoom:100, near:.1, far:40 }}
    dpr={[1,1.5]} frameloop={!visible || reduced || phase === 'processing' ? 'demand' : 'always'}
    gl={{ alpha:true, antialias:true, powerPreference:'low-power' }} fallback={null}>
    <Orb anchor={anchor} phase={phase} reduced={reduced} onReady={onReady}/>
  </Canvas>
}
