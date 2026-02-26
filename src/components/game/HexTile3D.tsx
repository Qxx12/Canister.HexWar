import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import { useFrame } from '@react-three/fiber'
import type { ThreeEvent } from '@react-three/fiber'
import type { Tile, TerrainType } from '../../types/board'
import type { PlayerId } from '../../types/player'
import { axialToPixel } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import { getTerrainTexture } from './terrainTextures'

export const TILE_DEPTH = 5
export const SURFACE_Y = TILE_DEPTH / 2

// --- Tree placement ---

function seededRng(seed: number) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 4294967296
  }
}

interface TreeSpec { x: number; z: number; yRot: number; scale: number }

const TREE_TERRAINS = new Set<TerrainType>(['grassland', 'plains', 'tundra'])

function getTreeSpecs(terrain: TerrainType, q: number, r: number): TreeSpec[] {
  if (!TREE_TERRAINS.has(terrain)) return []
  const rng = seededRng((q * 127 + r * 311) >>> 0)
  const count =
    terrain === 'grassland' ? 1 + Math.floor(rng() * 3) :
    terrain === 'plains'    ? (rng() < 0.55 ? 1 : 0) :
    /* tundra */              1 + Math.floor(rng() * 2)
  return Array.from({ length: count }, () => {
    const angle = rng() * Math.PI * 2
    const dist  = 0.12 + rng() * 0.32
    return { x: Math.cos(angle) * dist, z: Math.sin(angle) * dist, yRot: rng() * Math.PI * 2, scale: 0.75 + rng() * 0.5 }
  })
}

const CANOPY_COLORS: Record<string, string> = {
  grassland: '#3a7d2c',
  plains:    '#6a8030',
  tundra:    '#8aaa8a',
}

function Trees({ terrain, hexSize, treeSpecs }: { terrain: TerrainType; hexSize: number; treeSpecs: TreeSpec[] }) {
  const isTundra = terrain === 'tundra'
  return (
    <>
      {treeSpecs.map((spec, i) => {
        const s = spec.scale
        const trunkH  = hexSize * 0.07 * s
        const coneH   = hexSize * 0.15 * s
        const coneR   = hexSize * (isTundra ? 0.05 : 0.07) * s
        const color   = CANOPY_COLORS[terrain]
        return (
          <group key={i} position={[spec.x * hexSize, SURFACE_Y, spec.z * hexSize]} rotation={[0, spec.yRot, 0]}>
            {/* Trunk */}
            <mesh position={[0, trunkH / 2, 0]} raycast={() => null}>
              <cylinderGeometry args={[hexSize * 0.018 * s, hexSize * 0.025 * s, trunkH, 5]} />
              <meshLambertMaterial color="#5c3a1e" />
            </mesh>
            {/* Shadow disc cast by upper layer onto lower */}
            <mesh position={[0, trunkH + coneH * 0.45, 0]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
              <circleGeometry args={[coneR * 0.82, 6]} />
              <meshLambertMaterial color="#0f2a0a" transparent opacity={0.45} />
            </mesh>
            {/* Stacked cone layers — each level narrower and higher */}
            {[0, 1].map(level => {
              const rScale = 1 - level * 0.28
              const hScale = 1 - level * 0.2
              const yBase  = trunkH + level * coneH * 0.45
              return (
                <mesh key={level} position={[0, yBase + (coneH * hScale) / 2, 0]} raycast={() => null}>
                  <coneGeometry args={[coneR * rScale, coneH * hScale, 6]} />
                  <meshLambertMaterial color={color} />
                </mesh>
              )
            })}
          </group>
        )
      })}
    </>
  )
}

// --- Oasis (desert only) ---

interface PalmSpec { angle: number; dist: number; scale: number; tilt: number; tiltDir: number }

function getOasisSpecs(q: number, r: number, poolR: number): PalmSpec[] | null {
  const rng = seededRng((q * 211 + r * 337) >>> 0)
  if (rng() > 0.06) return null
  const count = 1 + Math.floor(rng() * 2) // 1–2 palms
  return Array.from({ length: count }, () => ({
    angle:   rng() * Math.PI * 2,
    dist:    poolR * (0.7 + rng() * 0.2),
    scale:   0.8 + rng() * 0.4,
    tilt:    0.18 + rng() * 0.15,  // lean outward from pool
    tiltDir: rng() * Math.PI * 2,
  }))
}

function Oasis({ hexSize, q, r }: { hexSize: number; q: number; r: number }) {
  const rng = seededRng((q * 211 + r * 337) >>> 0)
  rng() // skip probability roll
  const ox = (rng() - 0.5) * hexSize * 0.25
  const oz = (rng() - 0.5) * hexSize * 0.25
  const poolR = hexSize * 0.48

  const palms = useMemo(() => getOasisSpecs(q, r, poolR) ?? [], [q, r, poolR])

  const rocks = useMemo(() => {
    const rockRng = seededRng((q * 457 + r * 293) >>> 0)
    const count = 4 + Math.floor(rockRng() * 5)
    return Array.from({ length: count }, () => {
      const angle = rockRng() * Math.PI * 2
      const dist  = poolR * (0.85 + rockRng() * 0.1)
      const w     = hexSize * (0.033 + rockRng() * 0.042)
      const h     = w * (0.35 + rockRng() * 0.25)
      const d     = w * (0.6 + rockRng() * 0.4)
      const yRot  = rockRng() * Math.PI * 2
      const shade = Math.floor(180 + rockRng() * 45)
      const color = `rgb(${shade},${Math.floor(shade * 0.88)},${Math.floor(shade * 0.68)})`
      return { x: Math.cos(angle) * dist, z: Math.sin(angle) * dist, w, h, d, yRot, color }
    })
  }, [q, r, poolR, hexSize])

  const waterTexture = useMemo(() => {
    const size = 128
    const canvas = document.createElement('canvas')
    canvas.width = size
    canvas.height = size
    const ctx = canvas.getContext('2d')!
    const c = size / 2
    const blobRng = seededRng((q * 173 + r * 401) >>> 0)

    // Organic blob: perturbed points around circle, smoothed with midpoint bezier
    const numPts = 8
    const pts: [number, number][] = Array.from({ length: numPts }, (_, i) => {
      const angle = (i / numPts) * Math.PI * 2
      const rad = c * (0.68 + blobRng() * 0.28)
      return [c + Math.cos(angle) * rad, c + Math.sin(angle) * rad]
    })

    ctx.beginPath()
    for (let i = 0; i < numPts; i++) {
      const curr = pts[i]
      const next = pts[(i + 1) % numPts]
      const mid: [number, number] = [(curr[0] + next[0]) / 2, (curr[1] + next[1]) / 2]
      if (i === 0) ctx.moveTo(mid[0], mid[1])
      else ctx.quadraticCurveTo(curr[0], curr[1], mid[0], mid[1])
    }
    // close with last bezier back to start
    const first = pts[0]
    const last  = pts[numPts - 1]
    const closeMid: [number, number] = [(last[0] + first[0]) / 2, (last[1] + first[1]) / 2]
    ctx.quadraticCurveTo(last[0], last[1], closeMid[0], closeMid[1])
    ctx.closePath()

    ctx.save()
    ctx.clip()
    const grad = ctx.createRadialGradient(c, c, 0, c, c, c)
    grad.addColorStop(0,   'rgba(28, 85, 68, 0.95)')
    grad.addColorStop(0.6, 'rgba(22, 68, 54, 0.80)')
    grad.addColorStop(1,   'rgba(16, 52, 42, 0)')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, size, size)
    ctx.restore()

    const tex = new THREE.CanvasTexture(canvas)
    tex.needsUpdate = true
    return tex
  }, [q, r])

  return (
    <group position={[ox, SURFACE_Y, oz]}>
      {/* Water pool */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.3, 0]} renderOrder={1} raycast={() => null}>
        <planeGeometry args={[poolR * 2, poolR * 2]} />
        <meshBasicMaterial map={waterTexture} transparent depthWrite={false} depthTest={false} />
      </mesh>
      {/* Rocks around pool */}
      {rocks.map((rock, i) => (
        <mesh key={i} position={[rock.x, 0.3 + rock.h / 2, rock.z]} rotation={[0, rock.yRot, 0]} scale={[rock.w, rock.h, rock.d]} renderOrder={2} raycast={() => null}>
          <sphereGeometry args={[1, 5, 4]} />
          <meshLambertMaterial color={rock.color} transparent />
        </mesh>
      ))}
      {/* Palm trees */}
      {palms.map((p, i) => {
        const s = p.scale
        const trunkH = hexSize * 0.30 * s
        // Tilt trunk away from pool center
        const leanX = Math.sin(p.tiltDir) * p.tilt
        const leanZ = Math.cos(p.tiltDir) * p.tilt
        return (
          <group key={i} position={[Math.cos(p.angle) * p.dist, 0, Math.sin(p.angle) * p.dist]}>
            {/* Tilted group — trunk and fronds share the same lean */}
            <group rotation={[leanX, 0, leanZ]}>
              {/* Trunk */}
              <mesh position={[0, trunkH / 2, 0]} raycast={() => null}>
                <cylinderGeometry args={[hexSize * 0.01 * s, hexSize * 0.018 * s, trunkH, 5]} />
                <meshLambertMaterial color="#8b6914" />
              </mesh>
              {/* Fronds — always at trunk tip in local space */}
              <group position={[0, trunkH, 0]}>
                {Array.from({ length: 11 }, (_, j) => {
                  const frondL = hexSize * 0.14 * s
                  const frondW = hexSize * 0.022 * s
                  const droop = j < 2 ? -Math.PI * 0.12 : Math.PI * 0.28
                  return (
                    <group key={j} rotation={[0, (j / 11) * Math.PI * 2, 0]}>
                      <group rotation={[droop, 0, 0]}>
                        <mesh position={[0, 0, frondL / 2]} raycast={() => null}>
                          <boxGeometry args={[frondW, hexSize * 0.004 * s, frondL]} />
                          <meshLambertMaterial color="#4a8a20" />
                        </mesh>
                      </group>
                    </group>
                  )
                })}
              </group>
            </group>
          </group>
        )
      })}
    </group>
  )
}

// --- Deer (tundra only, very sparse) ---

function getDeerSpec(q: number, r: number): { x: number; z: number; yRot: number } | null {
  const rng = seededRng((q * 503 + r * 179) >>> 0)
  if (rng() > 0.25) return null
  return {
    x:    (rng() - 0.5) * 0.7,
    z:    (rng() - 0.5) * 0.7,
    yRot: rng() * Math.PI * 2,
  }
}

function Deer({ hexSize, spec }: { hexSize: number; spec: { x: number; z: number; yRot: number } }) {
  const s = hexSize * 0.1
  const bodyColor   = '#c4a882'
  const legColor    = '#a08060'
  const antlerColor = '#7a5c18'

  const neckRef = useRef<THREE.Mesh>(null)
  const headRef = useRef<THREE.Mesh>(null)
  const antlerGroupRef = useRef<THREE.Group>(null)
  const deerState = useRef({ slot: -1, grazing: false, startTime: 0 })

  // Unique time offset per deer to desync slot boundaries
  const phaseOffset = Math.abs(spec.x * 7.3 + spec.z * 3.7) % 8

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime()
    const slot = Math.floor((t + phaseOffset) / 8)

    // Each new 8s slot: independently roll whether this deer grazes (~8% chance)
    if (slot !== deerState.current.slot) {
      deerState.current.slot = slot
      const rng = seededRng(((slot * 7919 + Math.round(spec.x * 100) * 1031 + Math.round(spec.z * 100) * 3037) >>> 0))
      if (rng() < 0.08) {
        deerState.current.grazing = true
        deerState.current.startTime = t
      } else {
        deerState.current.grazing = false
      }
    }

    let graze = 0
    if (deerState.current.grazing) {
      const elapsed = t - deerState.current.startTime
      const duration = 3.2
      if (elapsed >= duration) {
        deerState.current.grazing = false
      } else if (elapsed < 0.8) {
        graze = elapsed / 0.8
      } else if (elapsed < duration - 0.8) {
        graze = 1
      } else {
        graze = 1 - (elapsed - (duration - 0.8)) / 0.8
      }
    }

    if (neckRef.current) {
      neckRef.current.rotation.x = 0.5 + graze * 0.9
      neckRef.current.position.y = s * 0.85 - graze * s * 0.28
      neckRef.current.position.z = s * 0.55 + graze * s * 0.18
    }
    if (headRef.current) {
      headRef.current.rotation.x = graze * 1.1
      headRef.current.position.y = s * 1.05 - graze * s * 0.62
      headRef.current.position.z = s * 0.75 + graze * s * 0.28
    }
    if (antlerGroupRef.current) {
      antlerGroupRef.current.position.y = s * 1.22 - graze * s * 0.62
      antlerGroupRef.current.position.z = s * 0.72 + graze * s * 0.28
      antlerGroupRef.current.rotation.x = graze * 1.1
    }
  })

  return (
    <group position={[spec.x * hexSize, SURFACE_Y, spec.z * hexSize]} rotation={[0, spec.yRot, 0]}>
      {/* Body */}
      <mesh position={[0, s * 0.55, 0]} raycast={() => null}>
        <boxGeometry args={[s * 0.5, s * 0.4, s * 1.0]} />
        <meshLambertMaterial color={bodyColor} />
      </mesh>
      {/* Neck */}
      <mesh ref={neckRef} position={[0, s * 0.85, s * 0.55]} rotation={[0.5, 0, 0]} raycast={() => null}>
        <boxGeometry args={[s * 0.22, s * 0.38, s * 0.18]} />
        <meshLambertMaterial color={bodyColor} />
      </mesh>
      {/* Head */}
      <mesh ref={headRef} position={[0, s * 1.05, s * 0.75]} raycast={() => null}>
        <boxGeometry args={[s * 0.26, s * 0.26, s * 0.36]} />
        <meshLambertMaterial color={bodyColor} />
      </mesh>
      {/* Legs */}
      {([[-0.16, -0.32], [0.16, -0.32], [-0.16, 0.32], [0.16, 0.32]] as [number, number][]).map(([lx, lz], i) => (
        <mesh key={i} position={[lx * s, s * 0.18, lz * s]} raycast={() => null}>
          <boxGeometry args={[s * 0.1, s * 0.44, s * 0.1]} />
          <meshLambertMaterial color={legColor} />
        </mesh>
      ))}
      {/* Antlers */}
      <group ref={antlerGroupRef} position={[0, s * 1.22, s * 0.72]}>
        {([-1, 1] as const).map((side, i) => (
          <group key={i} position={[side * s * 0.1, 0, 0]}>
            <mesh position={[side * s * 0.1, s * 0.22, 0]} rotation={[0, 0, side * 0.45]} raycast={() => null}>
              <boxGeometry args={[s * 0.05, s * 0.38, s * 0.05]} />
              <meshLambertMaterial color={antlerColor} />
            </mesh>
            <mesh position={[side * s * 0.22, s * 0.38, -s * 0.1]} rotation={[0.3, 0, side * 0.9]} raycast={() => null}>
              <boxGeometry args={[s * 0.04, s * 0.22, s * 0.04]} />
              <meshLambertMaterial color={antlerColor} />
            </mesh>
          </group>
        ))}
      </group>
    </group>
  )
}

// --- Main tile component ---

interface HexTile3DProps {
  tile: Tile
  hexSize: number
  playerIndex: (id: PlayerId) => number
  isSelected: boolean
  isValidDestination: boolean
  onClick: () => void
  onPointerOver: (e: ThreeEvent<PointerEvent>) => void
  onPointerOut: () => void
}

export function HexTile3D({
  tile, hexSize, playerIndex, isSelected, isValidDestination,
  onClick, onPointerOver, onPointerOut,
}: HexTile3DProps) {
  const { x, y: z } = axialToPixel(tile.coord, hexSize)

  const terrainTexture = useMemo(() => getTerrainTexture(tile.terrain), [tile.terrain])
  const topColor = useMemo(() => new THREE.Color(isSelected ? '#aaaaaa' : '#ffffff'), [isSelected])
  const treeSpecs = useMemo(
    () => getTreeSpecs(tile.terrain, tile.coord.q, tile.coord.r),
    [tile.terrain, tile.coord.q, tile.coord.r]
  )
  const hasOasis = useMemo(
    () => tile.terrain === 'desert' && getOasisSpecs(tile.coord.q, tile.coord.r, hexSize * 0.1) !== null,
    [tile.terrain, tile.coord.q, tile.coord.r, hexSize]
  )
  const deerSpec = useMemo(
    () => tile.terrain === 'tundra' ? getDeerSpec(tile.coord.q, tile.coord.r) : null,
    [tile.terrain, tile.coord.q, tile.coord.r]
  )

  const playerColor = tile.owner !== null ? PLAYER_COLORS[playerIndex(tile.owner)] : '#666'
  const starColor = tile.startOwner ? PLAYER_COLORS[playerIndex(tile.startOwner)] : '#666'

  return (
    <group position={[x, 0, z]}>
      <mesh
        rotation={[0, Math.PI / 3, 0]}
        onClick={e => { e.stopPropagation(); onClick() }}
        onPointerOver={onPointerOver}
        onPointerMove={onPointerOver}
        onPointerOut={onPointerOut}
      >
        <cylinderGeometry args={[hexSize, hexSize, TILE_DEPTH, 6]} />
        <meshLambertMaterial attach="material-0" color="#c2a97a" />
        <meshLambertMaterial
          attach="material-1"
          map={terrainTexture}
          color={topColor}
          emissive={isValidDestination ? '#1db954' : '#000000'}
          emissiveIntensity={isValidDestination ? 0.35 : 0}
        />
        <meshLambertMaterial attach="material-2" color="#c2a97a" />
      </mesh>
      <Trees terrain={tile.terrain} hexSize={hexSize} treeSpecs={treeSpecs} />
      {hasOasis && <Oasis hexSize={hexSize} q={tile.coord.q} r={tile.coord.r} />}
      {deerSpec && <Deer hexSize={hexSize} spec={deerSpec} />}
      {tile.owner !== null && (
        <Html center position={[0, SURFACE_Y + 2, 0]} zIndexRange={[9, 0]} pointerEvents="none" style={{ pointerEvents: 'none' }}>
          <div style={{ textAlign: 'center', userSelect: 'none', whiteSpace: 'nowrap', position: 'relative' }}>
            {tile.isStartTile && (
              <div style={{
                position: 'absolute',
                bottom: '130%',
                left: '50%',
                transform: 'translateX(-50%)',
                color: starColor,
                fontSize: '12px',
                lineHeight: 1,
                textShadow: '-1px 0 white, 0 1px white, 1px 0 white, 0 -1px white',
              }}>★</div>
            )}
            <div style={{
              color: playerColor,
              fontWeight: 'bold',
              fontSize: '14px',
              lineHeight: 1,
              textShadow: '-1px 0 white, 0 1px white, 1px 0 white, 0 -1px white',
            }}>
              {tile.units}
            </div>
          </div>
        </Html>
      )}
    </group>
  )
}
