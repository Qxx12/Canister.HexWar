import type { MovementOrder } from '../../types/orders'
import type { Board } from '../../types/board'
import { axialToPixel } from '../../types/hex'
import styles from './MovementArrow.module.scss'

interface MovementArrowProps {
  order: MovementOrder
  board: Board
  hexSize: number
  isClamped: boolean
}

export function MovementArrow({ order, board, hexSize, isClamped }: MovementArrowProps) {
  const fromTile = board.get(order.fromKey)
  const toTile = board.get(order.toKey)
  if (!fromTile || !toTile) return null

  const from = axialToPixel(fromTile.coord, hexSize)
  const to = axialToPixel(toTile.coord, hexSize)

  // Midpoint between centers (at the shared edge)
  const mx = (from.x + to.x) / 2
  const my = (from.y + to.y) / 2

  // Direction vector
  const dx = to.x - from.x
  const dy = to.y - from.y
  const len = Math.sqrt(dx * dx + dy * dy)
  const nx = dx / len
  const ny = dy / len

  // Arrow length
  const arrowLen = hexSize * 0.35
  const x1 = mx - nx * arrowLen / 2
  const y1 = my - ny * arrowLen / 2
  const x2 = mx + nx * arrowLen / 2
  const y2 = my + ny * arrowLen / 2

  const color = isClamped ? '#c0392b' : '#111111'

  // Arrowhead as a triangle polygon, rotated to face destination
  const angleRad = Math.atan2(ny, nx)
  const angleDeg = (angleRad * 180) / Math.PI
  const tipSize = hexSize * 0.22
  // Triangle: tip at (tipSize, 0), base at (-tipSize/2, ±tipSize*0.6) in local space
  const tipPoints = `${tipSize},0 ${-tipSize * 0.5},${tipSize * 0.6} ${-tipSize * 0.5},${-tipSize * 0.6}`

  return (
    <g className={styles.arrow}>
      <polygon
        points={tipPoints}
        fill={color}
        stroke="white"
        strokeWidth={1.5}
        strokeLinejoin="round"
        transform={`translate(${x2}, ${y2}) rotate(${angleDeg})`}
      />
      <text
        x={x1}
        y={y1}
        textAnchor="middle"
        dominantBaseline="middle"
        className={styles.label}
        fill={color}
        stroke="white"
        strokeWidth={2.5}
        paintOrder="stroke fill"
        fontSize={hexSize * 0.28}
      >
        {order.requestedUnits}
      </text>
    </g>
  )
}
