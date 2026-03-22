import { useState, useMemo, useCallback, useEffect, useRef } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Line } from '@react-three/drei'
import * as THREE from 'three'
import type { ThreeEvent } from '@react-three/fiber'
import type { GameState } from '@hexwar/engine'
import type { AnimationEvent } from '@hexwar/engine'
import type { Board, Tile } from '@hexwar/engine'
import type { PlayerId } from '@hexwar/engine'
import { HexTile3D, SURFACE_Y } from './HexTile3D'
import { MovementArrow3D } from './MovementArrow3D'
import { AnimationLayer3D } from './AnimationLayer3D'
import { OrderModal } from './OrderModal'
import { TileTooltip } from './TileTooltip'
import { axialToPixel, hexCorners, hexNeighbors, hexToKey } from '@hexwar/engine'
import { PLAYER_COLORS } from '@hexwar/engine'

// Directional light placed at the specular mirror of the camera's default position
// (same height, same radial distance, opposite Z) so sun glints off flat pool surfaces
function SunLight({ boardRadius, shadowsEnabled }: { boardRadius: number; shadowsEnabled: boolean }) {
  return (
    <directionalLight
      castShadow={shadowsEnabled}
      position={[boardRadius * 0.8, boardRadius * 1.5, -boardRadius * 0.85]}
      intensity={1.8}
      shadow-mapSize={[2048, 2048]}
      shadow-camera-left={-boardRadius * 1.5}
      shadow-camera-right={boardRadius * 1.5}
      shadow-camera-top={boardRadius * 1.5}
      shadow-camera-bottom={-boardRadius * 1.5}
      shadow-camera-near={1}
      shadow-camera-far={boardRadius * 6}
    />
  )
}

// Maps zoom level directly to view angle — lower angle when zoomed in
function CameraTiltController({ minDist, maxDist }: { minDist: number; maxDist: number }) {
  const { camera } = useThree()
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controls = useThree(state => state.controls) as any
  const prevDist = useRef<number | null>(null)

  useFrame(() => {
    if (!controls?.target) return
    const target: THREE.Vector3 = controls.target
    const offset = new THREE.Vector3().subVectors(camera.position, target)
    const dist = offset.length()

    // Only correct polar angle while the user is actively zooming
    const prev = prevDist.current
    prevDist.current = dist
    if (prev === null || Math.abs(dist - prev) < 0.01) return

    const t = THREE.MathUtils.clamp((maxDist - dist) / (maxDist - minDist), 0, 1)
    const polar = THREE.MathUtils.lerp(Math.PI * 0.17, Math.PI * 0.28, t)

    const azimuth = Math.atan2(offset.x, offset.z)
    camera.position.set(
      target.x + dist * Math.sin(polar) * Math.sin(azimuth),
      target.y + dist * Math.cos(polar),
      target.z + dist * Math.sin(polar) * Math.cos(azimuth),
    )
    camera.lookAt(target)
  }, -1)

  return null
}

interface GameBoard3DProps {
  gameState: GameState
  activeAnimation: AnimationEvent | null
  arrowBoard: Board
  sunEnabled: boolean
  shadowsEnabled: boolean
  onSetOrder: (fromKey: string, toKey: string, units: number) => void
  onCancelOrder: (fromKey: string) => void
  onSetStandingOrder: (fromKey: string, toKey: string, units: number) => void
  onCancelStandingOrder: (fromKey: string) => void
}

const HEX_SIZE = 36
const BORDER_Y = SURFACE_Y + 0.5
const GRID_Y = SURFACE_Y + 0.1


const STRIP_VERT = `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`
const STRIP_FRAG = `
  uniform vec3 uColor;
  varying vec2 vUv;
  void main() {
    float edge = abs(vUv.y - 0.5) * 2.0;
    float alpha = pow(1.0 - edge, 3.5);
    gl_FragColor = vec4(uColor, alpha);
  }
`

const STRIP_HALF_WIDTH = 3.5

type Vec2 = [number, number]
type Quad = { key: string; color: string; v: [[number,number,number],[number,number,number],[number,number,number],[number,number,number]] }

// Returns the perpendicular to segment (p1→p2) pointing AWAY from (tileCx, tileCz)
function outwardPerp(p1: Vec2, p2: Vec2, tileCx: number, tileCz: number): Vec2 {
  const dx = p2[0] - p1[0], dz = p2[1] - p1[1]
  const len = Math.sqrt(dx * dx + dz * dz)
  if (len === 0) return [0, 1]
  // Two candidate perps
  const a: Vec2 = [-dz / len, dx / len]
  const b: Vec2 = [dz / len, -dx / len]
  const mx = (p1[0] + p2[0]) / 2, mz = (p1[1] + p2[1]) / 2
  const toTile = [tileCx - mx, tileCz - mz]
  // Pick the one pointing AWAY from tile center
  return (a[0] * toTile[0] + a[1] * toTile[1]) < 0 ? a : b
}

function buildBorderQuads(
  rawSegs: Array<{ key: string; p1: Vec2; p2: Vec2; color: string; tileCx: number; tileCz: number }>
): Quad[] {
  const vKey = (p: Vec2) => `${p[0].toFixed(1)},${p[1].toFixed(1)}`

  // For each vertex: collect all outward perps from segments touching it
  type VEntry = { outPerp: Vec2; segKey: string; isP1: boolean }
  const vertexEntries = new Map<string, VEntry[]>()
  for (const seg of rawSegs) {
    const op = outwardPerp(seg.p1, seg.p2, seg.tileCx, seg.tileCz)
    const push = (k: string, e: VEntry) => { const a = vertexEntries.get(k) ?? []; a.push(e); vertexEntries.set(k, a) }
    push(vKey(seg.p1), { outPerp: op, segKey: seg.key, isP1: true })
    push(vKey(seg.p2), { outPerp: op, segKey: seg.key, isP1: false })
  }

  return rawSegs.flatMap(seg => {
    const W = STRIP_HALF_WIDTH
    const op = outwardPerp(seg.p1, seg.p2, seg.tileCx, seg.tileCz)

    const miterAt = (vk: string, myKey: string): Vec2 => {
      const entries = (vertexEntries.get(vk) ?? []).filter(e => e.segKey !== myKey)
      const rawAdj = entries[0]?.outPerp ?? null
      const px = op[0], pz = op[1]
      if (!rawAdj) return px === 0 && pz === 0 ? [0, 0] : [px * W, pz * W]
      // Align adjacent perp to the same half-space as our outward perp
      const dot0 = rawAdj[0] * px + rawAdj[1] * pz
      const adjPerp: Vec2 = dot0 < 0 ? [-rawAdj[0], -rawAdj[1]] : rawAdj
      const mx = px + adjPerp[0], mz = pz + adjPerp[1]
      const len = Math.sqrt(mx * mx + mz * mz)
      if (len < 0.001) return [px * W, pz * W]
      const mdx = mx / len, mdz = mz / len
      const dot = mdx * px + mdz * pz
      const miterLen = dot > 0.15 ? Math.min(W / dot, W * 2.5) : W
      return [mdx * miterLen, mdz * miterLen]
    }

    const [ox1, oz1] = miterAt(vKey(seg.p1), seg.key)
    const [ox2, oz2] = miterAt(vKey(seg.p2), seg.key)

    return [{
      key: seg.key, color: seg.color,
      v: [
        [seg.p1[0] + ox1, BORDER_Y, seg.p1[1] + oz1],
        [seg.p1[0] - ox1, BORDER_Y, seg.p1[1] - oz1],
        [seg.p2[0] + ox2, BORDER_Y, seg.p2[1] + oz2],
        [seg.p2[0] - ox2, BORDER_Y, seg.p2[1] - oz2],
      ] as Quad['v'],
    }]
  })
}

function BorderStrip({ quad, color }: { quad: Quad['v']; color: string }) {
  const geometry = useMemo(() => {
    const [v0, v1, v2, v3] = quad
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array([...v0,...v1,...v2,...v3]), 3))
    geo.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0,1, 0,0, 1,1, 1,0]), 2))
    geo.setIndex([0, 1, 2, 2, 1, 3])
    return geo
  }, [quad])

  const matRef = useRef<THREE.ShaderMaterial>(null)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const uniforms = useMemo(() => ({ uColor: { value: new THREE.Color(color) } }), [])
  useEffect(() => {
    if (matRef.current) matRef.current.uniforms.uColor.value.set(color)
  }, [color])

  return (
    <mesh geometry={geometry} raycast={() => null}>
      <shaderMaterial
        ref={matRef}
        vertexShader={STRIP_VERT}
        fragmentShader={STRIP_FRAG}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        side={THREE.DoubleSide}
      />
    </mesh>
  )
}

function HexGrid3D({ board }: { board: Board }) {
  const edges = useMemo(() => {
    const seen = new Set<string>()
    const result: Array<{ key: string; points: [number, number, number][] }> = []

    for (const [key, tile] of board) {
      const { x: cx, y: cy } = axialToPixel(tile.coord, HEX_SIZE)
      const corners = hexCorners(cx, cy, HEX_SIZE)
      const neighbors = hexNeighbors(tile.coord)
      for (let j = 0; j < 6; j++) {
        const nKey = hexToKey(neighbors[j])
        const edgeKey = [key, nKey].sort().join('|')
        if (seen.has(edgeKey)) continue
        seen.add(edgeKey)
        const c1 = corners[(5 - j + 6) % 6]
        const c2 = corners[(6 - j + 6) % 6]
        result.push({ key: edgeKey, points: [[c1.x, GRID_Y, c1.y], [c2.x, GRID_Y, c2.y]] })
      }
    }
    return result
  }, [board])

  return (
    <>
      {edges.map(({ key, points }) => (
        <Line key={key} points={points} color="#999999" lineWidth={1} raycast={() => null} />
      ))}
    </>
  )
}


function TerritoryBorders3D({ board, playerIndex }: { board: Board; playerIndex: (id: PlayerId) => number }) {
  const quads = useMemo(() => {
    type Seg = { c1: { x: number; y: number }; c2: { x: number; y: number }; color: string; tileCx: number; tileCz: number }
    const edgeMap = new Map<string, Seg[]>()

    for (const [key, tile] of board) {
      if (tile.owner === null) continue
      const { x: cx, y: cy } = axialToPixel(tile.coord, HEX_SIZE)
      const corners = hexCorners(cx, cy, HEX_SIZE)
      const color = PLAYER_COLORS[playerIndex(tile.owner)]
      const neighbors = hexNeighbors(tile.coord)
      for (let j = 0; j < 6; j++) {
        const nKey = hexToKey(neighbors[j])
        const neighbor = board.get(nKey)
        if (!neighbor || neighbor.owner !== tile.owner) {
          const c1 = corners[(5 - j + 6) % 6]
          const c2 = corners[(6 - j + 6) % 6]
          const edgeKey = [key, nKey].sort().join('|')
          const segs = edgeMap.get(edgeKey) ?? []
          segs.push({ c1, c2, color, tileCx: cx, tileCz: cy })
          edgeMap.set(edgeKey, segs)
        }
      }
    }

    const rawSegs: Array<{ key: string; p1: Vec2; p2: Vec2; color: string; tileCx: number; tileCz: number }> = []
    for (const [edgeKey, segs] of edgeMap) {
      const { c1, c2 } = segs[0]
      if (segs.length === 1) {
        rawSegs.push({ key: edgeKey, p1: [c1.x, c1.y], p2: [c2.x, c2.y], color: segs[0].color, tileCx: segs[0].tileCx, tileCz: segs[0].tileCz })
      } else {
        const mx = (c1.x + c2.x) / 2, mz = (c1.y + c2.y) / 2
        rawSegs.push({ key: `${edgeKey}-a`, p1: [c1.x, c1.y], p2: [mx, mz], color: segs[0].color, tileCx: segs[0].tileCx, tileCz: segs[0].tileCz })
        rawSegs.push({ key: `${edgeKey}-b`, p1: [mx, mz], p2: [c2.x, c2.y], color: segs[1].color, tileCx: segs[1].tileCx, tileCz: segs[1].tileCz })
      }
    }

    return buildBorderQuads(rawSegs)
  }, [board, playerIndex])

  return (
    <>
      {quads.map(({ key, v, color }) => (
        <BorderStrip key={key} quad={v} color={color} />
      ))}
    </>
  )
}

function Scene({
  gameState, activeAnimation, arrowBoard,
  selectedKey, validDestinations, sunEnabled, shadowsEnabled, boardRadius,
  onTileClick, onTilePointerOver, onTilePointerOut, playerIndex,
}: {
  gameState: GameState
  activeAnimation: AnimationEvent | null
  arrowBoard: Board
  selectedKey: string | null
  validDestinations: Set<string>
  sunEnabled: boolean
  shadowsEnabled: boolean
  boardRadius: number
  onTileClick: (key: string) => void
  onTilePointerOver: (tile: Tile, e: ThreeEvent<PointerEvent>) => void
  onTilePointerOut: () => void
  playerIndex: (id: PlayerId) => number
}) {
  const { board, humanPlayerId, orders, humanStandingOrders, phase } = gameState
  const isPlayerTurn = phase === 'playerTurn'
  const humanOrders = orders.get(humanPlayerId) ?? new Map()
  const tiles = Array.from(board.values())

  return (
    <>
      <ambientLight intensity={sunEnabled ? 0.9 : 1.5} />
      {sunEnabled && <SunLight boardRadius={boardRadius} shadowsEnabled={shadowsEnabled} />}
      {tiles.map(tile => {
        const key = hexToKey(tile.coord)
        return (
          <HexTile3D
            key={key}
            tile={tile}
            hexSize={HEX_SIZE}
            playerIndex={playerIndex}
            isSelected={selectedKey === key}
            isValidDestination={validDestinations.has(key)}
            onClick={() => onTileClick(key)}
            onPointerOver={e => onTilePointerOver(tile, e)}
            onPointerOut={onTilePointerOut}
          />
        )
      })}
      <HexGrid3D board={board} />
      <TerritoryBorders3D board={board} playerIndex={playerIndex} />
      {isPlayerTurn && Array.from(humanOrders.entries()).map(([fromKey, order]) => {
        const fromTileForArrow = arrowBoard.get(fromKey)
        const isClamped = fromTileForArrow
          ? order.requestedUnits !== Infinity && fromTileForArrow.units < order.requestedUnits
          : false
        const hollow = humanStandingOrders.has(fromKey)
        return (
          <MovementArrow3D
            key={fromKey}
            order={order}
            board={board}
            arrowBoard={arrowBoard}
            hexSize={HEX_SIZE}
            isClamped={isClamped}
            hollow={hollow}
          />
        )
      })}
      <AnimationLayer3D
        activeEvent={activeAnimation}
        board={board}
        hexSize={HEX_SIZE}
        playerIndex={playerIndex}
      />
    </>
  )
}

export function GameBoard3D({
  gameState, activeAnimation, arrowBoard, sunEnabled, shadowsEnabled,
  onSetOrder, onCancelOrder, onSetStandingOrder, onCancelStandingOrder,
}: GameBoard3DProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [pendingOrder, setPendingOrder] = useState<{ fromKey: string; toKey: string } | null>(null)
  const [tooltip, setTooltip] = useState<{ tile: Tile; x: number; y: number } | null>(null)
  const tooltipTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pendingTileRef = useRef<Tile | null>(null)
  const mousePosRef = useRef({ x: 0, y: 0 })

  const { board, players, humanPlayerId, orders, humanStandingOrders, phase } = gameState
  const isPlayerTurn = phase === 'playerTurn'

  const playerIndex = useCallback((id: PlayerId) => {
    return players.findIndex(p => p.id === id)
  }, [players])

  // Compute board center and radius for camera setup
  const { boardCenter, boardRadius } = useMemo(() => {
    let sumX = 0, sumZ = 0, count = 0, maxDist = 0
    const positions: { x: number; z: number }[] = []
    for (const tile of board.values()) {
      const { x, y } = axialToPixel(tile.coord, HEX_SIZE)
      sumX += x; sumZ += y; count++
      positions.push({ x, z: y })
    }
    const cx = count > 0 ? sumX / count : 0
    const cz = count > 0 ? sumZ / count : 0
    for (const p of positions) {
      const d = Math.sqrt((p.x - cx) ** 2 + (p.z - cz) ** 2)
      maxDist = Math.max(maxDist, d)
    }
    return { boardCenter: { x: cx, z: cz }, boardRadius: maxDist + HEX_SIZE }
  }, [board])

  const validDestinations = useMemo(() => {
    if (!selectedKey) return new Set<string>()
    const tile = board.get(selectedKey)
    if (!tile) return new Set<string>()
    return new Set(hexNeighbors(tile.coord).map(hexToKey).filter(k => board.has(k)))
  }, [selectedKey, board])

  const showTooltipAfterDelay = useCallback((tile: Tile) => {
    if (tooltipTimerRef.current) clearTimeout(tooltipTimerRef.current)
    pendingTileRef.current = tile
    tooltipTimerRef.current = setTimeout(() => {
      setTooltip({ tile: pendingTileRef.current!, ...mousePosRef.current })
    }, 500)
  }, [])

  const hideTooltip = useCallback(() => {
    if (tooltipTimerRef.current) clearTimeout(tooltipTimerRef.current)
    pendingTileRef.current = null
    setTooltip(null)
  }, [])

  const resetTooltipOnMove = useCallback(() => {
    setTooltip(null)
    if (!pendingTileRef.current) return
    if (tooltipTimerRef.current) clearTimeout(tooltipTimerRef.current)
    tooltipTimerRef.current = setTimeout(() => {
      if (pendingTileRef.current) {
        setTooltip({ tile: pendingTileRef.current, ...mousePosRef.current })
      }
    }, 500)
  }, [])

  useEffect(() => {
    if (!tooltip) return
    const current = board.get(hexToKey(tooltip.tile.coord))
    if (!current || current.units !== tooltip.tile.units || current.owner !== tooltip.tile.owner) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      hideTooltip()
    }
  }, [board, tooltip, hideTooltip])

  const handleTileClick = useCallback((key: string) => {
    if (!isPlayerTurn) return
    const tile = board.get(key)
    if (!tile) return

    if (!selectedKey) {
      if (tile.owner === humanPlayerId && tile.units > 0) setSelectedKey(key)
      return
    }
    if (selectedKey === key) { setSelectedKey(null); return }
    if (validDestinations.has(key)) {
      const currentOrders = orders.get(humanPlayerId) ?? new Map()
      if (currentOrders.get(key)?.toKey === selectedKey) { setSelectedKey(null); return }
      setPendingOrder({ fromKey: selectedKey, toKey: key })
      setSelectedKey(null)
    } else if (tile.owner === humanPlayerId && tile.units > 0) {
      setSelectedKey(key)
    } else {
      setSelectedKey(null)
    }
  }, [isPlayerTurn, board, selectedKey, humanPlayerId, validDestinations, orders])

  const handleTilePointerOver = useCallback((tile: Tile, e: ThreeEvent<PointerEvent>) => {
    mousePosRef.current = { x: e.clientX, y: e.clientY }
    showTooltipAfterDelay(tile)
  }, [showTooltipAfterDelay])

  const humanOrders = orders.get(humanPlayerId) ?? new Map()
  const fromTile = pendingOrder ? board.get(pendingOrder.fromKey) : null
  const maxUnits = fromTile?.units ?? 0
  const existingOrder = pendingOrder
    ? (humanOrders.get(pendingOrder.fromKey)?.requestedUnits ?? null)
    : null
  const isStanding = pendingOrder ? humanStandingOrders.has(pendingOrder.fromKey) : false

  const handleOrderConfirm = (units: number, standing: boolean) => {
    if (pendingOrder) {
      onSetOrder(pendingOrder.fromKey, pendingOrder.toKey, units)
      if (standing) {
        onSetStandingOrder(pendingOrder.fromKey, pendingOrder.toKey, units)
      } else {
        onCancelStandingOrder(pendingOrder.fromKey)
      }
      setPendingOrder(null)
    }
  }

  const handleOrderCancel = () => {
    if (pendingOrder) {
      onCancelOrder(pendingOrder.fromKey)
      onCancelStandingOrder(pendingOrder.fromKey)
      setPendingOrder(null)
    }
  }

  // Offset all 3D objects so the board is centered at the origin
  const groupOffset: [number, number, number] = [-boardCenter.x, 0, -boardCenter.z]

  return (
    <div
      style={{ width: '100%', height: '100%', position: 'relative', background: '#7e7368' }}
      onPointerMove={e => {
        mousePosRef.current = { x: e.clientX, y: e.clientY }
        resetTooltipOnMove()
      }}
    >
      {/* Noise texture behind the canvas, matching the 2D board */}
      <div style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        opacity: 0.25,
        backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='256' height='256'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='256' height='256' filter='url(%23n)'/%3E%3C/svg%3E\")",
        backgroundSize: '256px 256px',
      }} />
      <Canvas
        shadows
        camera={{ fov: 50 }}
        gl={{ alpha: true }}
        style={{ position: 'relative' }}
        onCreated={({ camera }) => {
          camera.position.set(0, boardRadius * 1.5, boardRadius * 0.85)
          camera.lookAt(0, 0, 0)
        }}
      >
        <OrbitControls
          makeDefault
          minDistance={boardRadius * 0.5}
          maxDistance={boardRadius * 1.6}
          enableDamping
          dampingFactor={0.08}
          enableRotate={false}
          screenSpacePanning={false}
          zoomSpeed={3}
          mouseButtons={{ LEFT: undefined as unknown as THREE.MOUSE, MIDDLE: undefined as unknown as THREE.MOUSE, RIGHT: THREE.MOUSE.PAN }}
        />
        <CameraTiltController minDist={boardRadius * 0.5} maxDist={boardRadius * 1.6} />
        <group position={groupOffset}>
          <Scene
            gameState={gameState}
            activeAnimation={activeAnimation}
            arrowBoard={arrowBoard}
            selectedKey={selectedKey}
            validDestinations={validDestinations}
            sunEnabled={sunEnabled}
            shadowsEnabled={shadowsEnabled}
            boardRadius={boardRadius}
            onTileClick={handleTileClick}
            onTilePointerOver={handleTilePointerOver}
            onTilePointerOut={hideTooltip}
            playerIndex={playerIndex}
          />
        </group>
      </Canvas>

      {tooltip && !pendingOrder && (
        <TileTooltip tile={tooltip.tile} players={players} x={tooltip.x} y={tooltip.y} />
      )}
      {pendingOrder && fromTile && maxUnits > 0 && (
        <OrderModal
          fromKey={pendingOrder.fromKey}
          toKey={pendingOrder.toKey}
          maxUnits={maxUnits}
          existingOrder={existingOrder}
          isStanding={isStanding}
          onConfirm={handleOrderConfirm}
          onCancel={handleOrderCancel}
          onClose={() => setPendingOrder(null)}
        />
      )}
    </div>
  )
}
