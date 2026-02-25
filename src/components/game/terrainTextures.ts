import * as THREE from 'three'
import type { TerrainType } from '../../types/board'

const SIZE = 256

function seededRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

function drawGrassland(ctx: CanvasRenderingContext2D, rng: () => number) {
  ctx.fillStyle = '#aed090'
  ctx.fillRect(0, 0, SIZE, SIZE)

  // Sparse grass tufts
  for (let i = 0; i < 10; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const len = 8 + rng() * 6
    for (let b = -1; b <= 1; b++) {
      const angle = -Math.PI / 2 + b * 0.3 + (rng() - 0.5) * 0.1
      ctx.beginPath()
      ctx.moveTo(x, y)
      ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len)
      ctx.strokeStyle = b === 0 ? 'rgba(114,194,74,0.55)' : 'rgba(45,110,28,0.45)'
      ctx.lineWidth = 1.5
      ctx.lineCap = 'round'
      ctx.stroke()
    }
  }

  // A couple of tiny flowers
  for (let i = 0; i < 3; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    ctx.beginPath()
    ctx.arc(x, y, 2, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(255,255,255,0.55)'
    ctx.fill()
  }
}

function drawTundra(ctx: CanvasRenderingContext2D, rng: () => number) {
  ctx.fillStyle = '#ddeaf4'
  ctx.fillRect(0, 0, SIZE, SIZE)

  // Subtle snow blobs
  for (let i = 0; i < 6; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const rx = 14 + rng() * 14
    const ry = rx * (0.5 + rng() * 0.3)
    ctx.beginPath()
    ctx.ellipse(x, y, rx, ry, rng() * Math.PI, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(240,248,255,${0.4 + rng() * 0.2})`
    ctx.fill()
  }

  // Bare soil patches — irregular polygons
  for (let i = 0; i < 8; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const r = 4 + rng() * 7
    const pts = 5 + Math.floor(rng() * 4)
    ctx.beginPath()
    for (let p = 0; p < pts; p++) {
      const a = (p / pts) * Math.PI * 2
      const pr = r * (0.5 + rng() * 0.5)
      const px = x + Math.cos(a) * pr
      const py = y + Math.sin(a) * pr * (0.5 + rng() * 0.5)
      p === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
    }
    ctx.closePath()
    ctx.fillStyle = `rgba(145, 118, 90, ${0.5 + rng() * 0.25})`
    ctx.fill()
  }

  // A few moss dots
  for (let i = 0; i < 12; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const r = 2 + rng() * 3.5
    ctx.beginPath()
    ctx.arc(x, y, r, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(72, 100, 48, ${0.3 + rng() * 0.25})`
    ctx.fill()
  }
}

function drawDesert(ctx: CanvasRenderingContext2D, rng: () => number) {
  ctx.fillStyle = '#e4d490'
  ctx.fillRect(0, 0, SIZE, SIZE)

  // A handful of pebbles
  for (let i = 0; i < 5; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const r = 2 + rng() * 3
    ctx.beginPath()
    ctx.ellipse(x, y, r, r * (0.5 + rng() * 0.4), rng() * Math.PI, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(168, 138, 75, ${0.4 + rng() * 0.2})`
    ctx.fill()
  }
}

function drawPlains(ctx: CanvasRenderingContext2D, rng: () => number) {
  ctx.fillStyle = '#c0d078'
  ctx.fillRect(0, 0, SIZE, SIZE)

  // Dry grass strokes
  for (let i = 0; i < 28; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    const len = 6 + rng() * 8
    const angle = -Math.PI / 2 + (rng() - 0.5) * 0.6
    ctx.beginPath()
    ctx.moveTo(x, y)
    ctx.lineTo(x + Math.cos(angle) * len, y + Math.sin(angle) * len)
    ctx.strokeStyle = `rgba(100, 72, 10, ${0.4 + rng() * 0.35})`
    ctx.lineWidth = 1.2
    ctx.lineCap = 'round'
    ctx.stroke()
  }

  // A couple of tiny flowers
  for (let i = 0; i < 3; i++) {
    const x = rng() * SIZE
    const y = rng() * SIZE
    ctx.beginPath()
    ctx.arc(x, y, 2, 0, Math.PI * 2)
    ctx.fillStyle = rng() > 0.5 ? 'rgba(210,70,40,0.6)' : 'rgba(220,190,30,0.6)'
    ctx.fill()
  }
}

const cache = new Map<TerrainType, THREE.CanvasTexture>()

export function getTerrainTexture(terrain: TerrainType): THREE.CanvasTexture {
  if (cache.has(terrain)) return cache.get(terrain)!

  const canvas = document.createElement('canvas')
  canvas.width = SIZE
  canvas.height = SIZE
  const ctx = canvas.getContext('2d')!
  const seed = terrain.split('').reduce((a, c) => a * 31 + c.charCodeAt(0), 0)
  const rng = seededRng(seed)

  switch (terrain) {
    case 'grassland': drawGrassland(ctx, rng); break
    case 'tundra':    drawTundra(ctx, rng);    break
    case 'desert':    drawDesert(ctx, rng);    break
    case 'plains':    drawPlains(ctx, rng);    break
  }

  const tex = new THREE.CanvasTexture(canvas)
  tex.needsUpdate = true
  cache.set(terrain, tex)
  return tex
}
