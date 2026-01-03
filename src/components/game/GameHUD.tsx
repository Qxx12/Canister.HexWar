import { useState } from 'react'
import type { GameState } from '../../types/game'
import { Button } from '../shared/Button'
import { Modal } from '../shared/Modal'
import styles from './GameHUD.module.scss'

interface GameHUDProps {
  gameState: GameState
  onEndTurn: () => void
  onRetire: () => void
  onZoomIn: () => void
  onZoomOut: () => void
  isAnimating: boolean
}

export function GameHUD({ gameState, onEndTurn, onRetire, onZoomIn, onZoomOut, isAnimating }: GameHUDProps) {
  const [showRetireConfirm, setShowRetireConfirm] = useState(false)
  const { phase, players, humanPlayerId, turn } = gameState
  const isPlayerTurn = phase === 'playerTurn'

  const statusText = isAnimating
    ? 'Animating...'
    : phase === 'playerTurn'
      ? 'Your Turn'
      : phase === 'aiTurn'
        ? 'AI Thinking...'
        : ''

  return (
    <>
      <div className={styles.hud}>
        <div className={styles.topBar}>
          <div className={styles.turnInfo}>
            <span className={styles.turnNumber}>Turn {turn.turnNumber}</span>
            <span className={`${styles.status} ${isPlayerTurn ? styles.playerTurn : ''}`}>
              {statusText}
            </span>
          </div>
          <div className={styles.controls}>
            <div className={styles.zoomControls}>
              <button className={styles.zoomBtn} onClick={onZoomOut} title="Zoom out">&#8722;</button>
              <button className={styles.zoomBtn} onClick={onZoomIn} title="Zoom in">&#43;</button>
            </div>
            <Button onClick={() => setShowRetireConfirm(true)} variant="secondary">
              Menu
            </Button>
          </div>
        </div>

        <div className={styles.legend}>
          {players.filter(p => !p.isEliminated).map(p => (
            <div key={p.id} className={styles.playerEntry}>
              <div className={styles.colorDot} style={{ background: p.color }} />
              <span className={`${styles.playerName} ${p.id === humanPlayerId ? styles.you : ''}`}>
                {p.name}{p.id === humanPlayerId ? ' (You)' : ''}
              </span>
            </div>
          ))}
        </div>

        {isPlayerTurn && !isAnimating && (
          <div className={styles.nextTurnWrapper}>
            <Button onClick={onEndTurn} variant="primary">
              End Turn
            </Button>
          </div>
        )}
      </div>

      {showRetireConfirm && (
        <Modal title="Menu" onClose={() => setShowRetireConfirm(false)}>
          <div className={styles.menuContent}>
            <p>Do you want to retire and return to the start screen?</p>
            <div className={styles.menuActions}>
              <Button onClick={onRetire} variant="danger">Retire</Button>
              <Button onClick={() => setShowRetireConfirm(false)} variant="secondary">Continue</Button>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
}
