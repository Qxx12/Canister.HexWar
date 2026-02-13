import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import type { MovementOrder } from '../../types/orders'
import type { Board } from '../../types/board'
import { axialToPixel } from '../../types/hex'
import { SURFACE_Y } from './HexTile3D'

const ARROW_Y = SURFACE_Y + 3

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

  const geom = useMemo(() => {
    if (!fromTile || !toTile) return null
    const from = axialToPixel(fromTile.coord, hexSize)
    const to = axialToPixel(toTile.coord, hexSize)
    const mx = (from.x + to.x) / 2
    const mz = (from.y + to.y) / 2
    const dx = to.x - from.x
    const dz = to.y - from.y
    const len = Math.sqrt(dx * dx + dz * dz)
    if (len === 0) return null
    const nx = dx / len
    const nz = dz / len
    const arrowLen = hexSize * 0.35
    const q = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(nx, 0, nz),
    )
    return {
      quaternion: q,
      midX: mx, midZ: mz,
      tipX: mx + nx * arrowLen / 2, tipZ: mz + nz * arrowLen / 2,
      labelX: mx - nx * arrowLen / 2, labelZ: mz - nz * arrowLen / 2,
      arrowLen,
    }
  }, [fromTile, toTile, hexSize])

  if (!fromTile || !toTile || !geom) return null

  const { quaternion, midX, midZ, tipX, tipZ, labelX, labelZ, arrowLen } = geom
  const color = isClamped ? '#c0392b' : '#111111'
  const opacity = hollow ? 0.5 : 1

  const displayUnits = order.requestedUnits === Infinity
    ? (arrowFromTile?.units ?? fromTile.units)
    : order.requestedUnits

  return (
    <group>
      <mesh position={[midX, ARROW_Y, midZ]} quaternion={quaternion}>
        <cylinderGeometry args={[1.2, 1.2, arrowLen * 0.6, 6]} />
        <meshBasicMaterial color={color} transparent opacity={opacity} />
      </mesh>
      <mesh position={[tipX, ARROW_Y, tipZ]} quaternion={quaternion}>
        <coneGeometry args={[hexSize * 0.09, hexSize * 0.13, 6]} />
        <meshBasicMaterial color={color} transparent opacity={opacity} />
      </mesh>
      <Html center position={[labelX, ARROW_Y + 6, labelZ]}>
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
