import { useState, useMemo, useCallback } from 'react'
import type { GameState } from '../../types/game'
import type { AnimationEvent } from '../../types/animation'
import type { PlayerId } from '../../types/player'
import { HexTile } from './HexTile'
import { MovementArrow } from './MovementArrow'
import { AnimationLayer } from './AnimationLayer'
import { OrderModal } from './OrderModal'
import { axialToPixel, hexNeighbors, hexToKey } from '../../types/hex'
import styles from './GameBoard.module.scss'

interface GameBoardProps {
  gameState: GameState
  activeAnimation: AnimationEvent | null
  viewport: { zoom: number; panX: number; panY: number }
  onPointerDown: (e: React.PointerEvent) => void
  onPointerMove: (e: React.PointerEvent) => void
  onPointerUp: (e: React.PointerEvent) => void
  onSetOrder: (fromKey: string, toKey: string, units: number) => void
  onCancelOrder: (fromKey: string) => void
}

const HEX_SIZE = 36

export function GameBoard({
  gameState,
  activeAnimation,
  viewport,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  onSetOrder,
  onCancelOrder,
}: GameBoardProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [pendingOrder, setPendingOrder] = useState<{ fromKey: string; toKey: string } | null>(null)

  const { board, players, humanPlayerId, orders, phase } = gameState
  const isPlayerTurn = phase === 'playerTurn'

  const playerIndex = useCallback((id: PlayerId) => {
    return players.findIndex(p => p.id === id)
  }, [players])

  const humanOrders = orders.get(humanPlayerId) ?? new Map()

  const { minX, minY, width, height } = useMemo(() => {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
    for (const tile of board.values()) {
      const { x, y } = axialToPixel(tile.coord, HEX_SIZE)
      minX = Math.min(minX, x - HEX_SIZE)
      minY = Math.min(minY, y - HEX_SIZE)
      maxX = Math.max(maxX, x + HEX_SIZE)
      maxY = Math.max(maxY, y + HEX_SIZE)
    }
    return { minX, minY, width: maxX - minX, height: maxY - minY }
  }, [board])

  const validDestinations = useMemo(() => {
    if (!selectedKey) return new Set<string>()
    const tile = board.get(selectedKey)
    if (!tile) return new Set<string>()
    const neighbors = hexNeighbors(tile.coord).map(hexToKey)
    return new Set(neighbors.filter(k => board.has(k)))
  }, [selectedKey, board])

  const handleTileClick = (key: string) => {
    if (!isPlayerTurn) return
    const tile = board.get(key)
    if (!tile) return

    if (!selectedKey) {
      if (tile.owner === humanPlayerId && tile.units > 0) {
        setSelectedKey(key)
      }
      return
    }

    if (selectedKey === key) {
      setSelectedKey(null)
      return
    }

    if (validDestinations.has(key)) {
      setPendingOrder({ fromKey: selectedKey, toKey: key })
      setSelectedKey(null)
    } else if (tile.owner === humanPlayerId && tile.units > 0) {
      setSelectedKey(key)
    } else {
      setSelectedKey(null)
    }
  }

  const fromTile = pendingOrder ? board.get(pendingOrder.fromKey) : null
  const maxUnits = fromTile?.units ?? 0
  const existingOrder = pendingOrder
    ? (humanOrders.get(pendingOrder.fromKey)?.requestedUnits ?? null)
    : null

  const handleOrderConfirm = (units: number) => {
    if (pendingOrder) {
      onSetOrder(pendingOrder.fromKey, pendingOrder.toKey, units)
      setPendingOrder(null)
    }
  }

  const handleOrderCancel = () => {
    if (pendingOrder) {
      onCancelOrder(pendingOrder.fromKey)
      setPendingOrder(null)
    }
  }

  const tiles = Array.from(board.values())

  return (
    <>
      <div
        className={styles.boardContainer}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      >
        <div
          className={styles.boardTransform}
          style={{
            transform: `translate(${viewport.panX}px, ${viewport.panY}px) scale(${viewport.zoom})`,
          }}
        >
          <svg
            width={width}
            height={height}
            viewBox={`${minX} ${minY} ${width} ${height}`}
            className={styles.svg}
          >
            <defs>
              <marker id="arrowhead-normal" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#111111" />
              </marker>
              <marker id="arrowhead-warn" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                <path d="M0,0 L0,6 L6,3 z" fill="#c0392b" />
              </marker>
            </defs>

            {tiles.map(tile => {
              const key = hexToKey(tile.coord)
              return (
                <HexTile
                  key={key}
                  tile={tile}
                  hexSize={HEX_SIZE}
                  playerIndex={playerIndex}
                  isSelected={selectedKey === key}
                  isValidDestination={validDestinations.has(key)}
                  onClick={() => handleTileClick(key)}
                />
              )
            })}

            {isPlayerTurn && Array.from(humanOrders.entries()).map(([fromKey, order]) => {
              const fromTileForArrow = board.get(fromKey)
              const isClamped = fromTileForArrow ? fromTileForArrow.units < order.requestedUnits : false
              return (
                <MovementArrow
                  key={fromKey}
                  order={order}
                  board={board}
                  hexSize={HEX_SIZE}
                  isClamped={isClamped}
                />
              )
            })}

            <AnimationLayer
              activeEvent={activeAnimation}
              board={board}
              hexSize={HEX_SIZE}
              playerIndex={playerIndex}
            />
          </svg>
        </div>
      </div>

      {pendingOrder && fromTile && maxUnits > 0 && (
        <OrderModal
          fromKey={pendingOrder.fromKey}
          toKey={pendingOrder.toKey}
          maxUnits={maxUnits}
          existingOrder={existingOrder}
          onConfirm={handleOrderConfirm}
          onCancel={handleOrderCancel}
          onClose={() => setPendingOrder(null)}
        />
      )}
    </>
  )
}
