import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import type { MovementOrder } from '../../types/orders'
import type { Board } from '../../types/board'
import { axialToPixel } from '../../types/hex'
import { SURFACE_Y } from './HexTile3D'

const ARROW_Y = SURFACE_Y + 1

interface MovementArrow3DProps {
  order: MovementOrder
  board: Board
  arrowBoard: Board
  hexSize: number
  isClamped: boolean
  hollow?: boolean
}

export function MovementArrow3D({
  order, board, arrowBoard, hexSize, isClamped, hollow = false,
}: MovementArrow3DProps) {
  const fromTile = board.get(order.fromKey)
  const toTile = board.get(order.toKey)
  const arrowFromTile = arrowBoard.get(order.fromKey)

  const { geom, labelX, labelZ } = useMemo(() => {
    if (!fromTile || !toTile) return { geom: null, labelX: 0, labelZ: 0 }
    const from = axialToPixel(fromTile.coord, hexSize)
    const to = axialToPixel(toTile.coord, hexSize)
    const dx = to.x - from.x
    const dz = to.y - from.y
    const len = Math.sqrt(dx * dx + dz * dz)
    if (len === 0) return { geom: null, labelX: 0, labelZ: 0 }
    const nx = dx / len, nz = dz / len
    const px = -nz, pz = nx   // perpendicular in XZ

    const mx = (from.x + to.x) / 2, mz = (from.y + to.y) / 2
    const hw = hexSize * 0.11   // half-width of triangle base
    const hl = hexSize * 0.34   // length (depth) of triangle
    // Arrow shifted toward destination, label stays near source side
    const arrowOffset = hexSize * 0.24
    const ax = mx + nx * arrowOffset, az = mz + nz * arrowOffset

    const Y = ARROW_Y
    const positions = new Float32Array([
      ax + px * hw - nx * hl / 2, Y, az + pz * hw - nz * hl / 2,
      ax - px * hw - nx * hl / 2, Y, az - pz * hw - nz * hl / 2,
      ax + nx * hl / 2,           Y, az + nz * hl / 2,
    ])
    const indices = new Uint16Array([0, 1, 2])
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setIndex(new THREE.BufferAttribute(indices, 1))

    return { geom: geo, labelX: mx - nx * hexSize * 0.15, labelZ: mz - nz * hexSize * 0.15 }
  }, [fromTile, toTile, hexSize])

  if (!fromTile || !toTile || !geom) return null

  const color = isClamped ? '#c0392b' : '#111111'
  const opacity = hollow ? 0.5 : 1

  const displayUnits = order.requestedUnits === Infinity
    ? (arrowFromTile?.units ?? fromTile.units)
    : order.requestedUnits

  return (
    <group>
      <mesh geometry={geom} raycast={() => null}>
        <meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} />
      </mesh>
      <Html center position={[labelX, ARROW_Y + 4, labelZ]} zIndexRange={[9, 0]} pointerEvents="none">
        <div style={{
          color,
          fontWeight: 'bold',
          fontSize: '13px',
          textShadow: '-1px 0 white, 0 1px white, 1px 0 white, 0 -1px white',
          pointerEvents: 'none',
          opacity,
          userSelect: 'none',
        }}>
          {displayUnits}
        </div>
      </Html>
    </group>
  )
}
