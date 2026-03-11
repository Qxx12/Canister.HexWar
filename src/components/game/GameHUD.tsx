import { useState, useEffect } from 'react'
import type { GameState } from '@hexwar/engine'
import { Modal } from '../shared/Modal'
import { Button } from '../shared/Button'
import styles from './GameHUD.module.scss'

interface GameHUDProps {
  gameState: GameState
  onEndTurn: () => void
  onRetire: () => void
  isAnimating: boolean
  animatingPlayerId: string | null
  viewMode: '2d' | '3d'
  onToggleView: () => void
  sunEnabled: boolean
  onToggleSun: () => void
  shadowsEnabled: boolean
  onToggleShadows: () => void
}

export function GameHUD({ gameState, onEndTurn, onRetire, isAnimating, animatingPlayerId, viewMode, onToggleView, sunEnabled, onToggleSun, shadowsEnabled, onToggleShadows }: GameHUDProps) {
  const [showRetireConfirm, setShowRetireConfirm] = useState(false)
  const { phase, players, humanPlayerId, turn } = gameState
  const isPlayerTurn = phase === 'playerTurn' && !isAnimating

  const activePlayerId = isAnimating
    ? animatingPlayerId
    : phase === 'playerTurn'
      ? humanPlayerId
      : null

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !showRetireConfirm) setShowRetireConfirm(true)
      if (e.key === 'Enter' && isPlayerTurn && !showRetireConfirm) onEndTurn()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [isPlayerTurn, showRetireConfirm, onEndTurn])

  return (
    <>
      <div className={styles.hud}>
        {/* Top bar: burger menu left, player list right */}
        <div className={styles.topBar}>
          <div className={styles.legend}>
            {players.filter(p => !p.isEliminated).map(p => (
              <div key={p.id} className={`${styles.playerEntry} ${p.id === activePlayerId ? styles.activeEntry : ''}`}>
                <div className={styles.colorDot} style={{ background: p.color }} />
                <span className={`${styles.playerName} ${p.id === humanPlayerId ? styles.you : ''}`}>
                  {p.name}{p.id === humanPlayerId ? ' (You)' : ''}
                </span>
              </div>
            ))}
          </div>
          <button className={styles.menuBtn} onClick={() => setShowRetireConfirm(true)} title="Menu">
            &#9776;
          </button>
        </div>

        {/* Bottom-left: zoom, end turn button, turn number */}
        <div className={styles.bottomLeft}>
          <div className={`${styles.endTurnWrap} ${!isPlayerTurn ? styles.waitingWrap : ''}`}>
            <button
              className={`${styles.endTurnBtn} ${!isPlayerTurn ? styles.waiting : ''}`}
              onClick={isPlayerTurn ? onEndTurn : undefined}
              disabled={!isPlayerTurn}
              title="End Turn"
            >
              {!isPlayerTurn ? <span className={styles.spinner} /> : <span className={styles.playIcon}>▶</span>}
            </button>
          </div>
          <span className={styles.turnNumber}>Turn {turn.turnNumber}</span>
        </div>
      </div>

      <div className={styles.viewToggle}>
        <button className={`${styles.viewToggleBtn} ${viewMode === '3d' ? styles.viewToggleBtnActive : ''}`} onClick={onToggleView}>
          {viewMode === '2d' ? '2D' : '3D'}
        </button>
        {viewMode === '3d' && (
          <button
            className={`${styles.viewToggleBtn} ${styles.sunBtn} ${sunEnabled ? styles.viewToggleBtnActive : ''}`}
            onClick={onToggleSun}
            title={sunEnabled ? 'Sun: on' : 'Sun: off'}
          >
            ☀
          </button>
        )}
        {viewMode === '3d' && (
          <button
            className={`${styles.viewToggleBtn} ${shadowsEnabled ? styles.viewToggleBtnActive : ''}`}
            onClick={onToggleShadows}
            title={shadowsEnabled ? 'Shadows: on' : 'Shadows: off'}
          >
            ◐
          </button>
        )}
      </div>

      {showRetireConfirm && (
        <Modal title="Menu" onClose={() => setShowRetireConfirm(false)} maxWidth={340}>
          <div className={styles.menuContent}>
            <p>Do you want to retire and return to the start screen?</p>
            <div className={styles.menuActions}>
              <Button onClick={onRetire} variant="danger">Retire</Button>
              <span style={{ marginLeft: 'auto' }}>
                <Button onClick={() => setShowRetireConfirm(false)} variant="secondary">Continue</Button>
              </span>
            </div>
          </div>
        </Modal>
      )}
    </>
  )
}
