import { useState, useEffect } from 'react'
import { Modal } from '../shared/Modal'
import { Button } from '../shared/Button'
import styles from './OrderModal.module.scss'

interface OrderModalProps {
  fromKey: string
  toKey: string
  maxUnits: number
  existingOrder: number | null
  isStanding: boolean
  onConfirm: (units: number, standing: boolean) => void
  onCancel: () => void
  onClose: () => void
}

export function OrderModal({ fromKey: _fromKey, toKey: _toKey, maxUnits, existingOrder, isStanding, onConfirm, onCancel, onClose }: OrderModalProps) {
  const [units, setUnits] = useState(existingOrder ?? Math.max(1, maxUnits))
  const [standing, setStanding] = useState(isStanding)

  useEffect(() => {
    setUnits(existingOrder ?? Math.max(1, maxUnits))
    setStanding(isStanding)
  }, [existingOrder, maxUnits, isStanding])

  const handleConfirm = () => {
    if (units >= 1 && units <= maxUnits) onConfirm(units, standing)
  }

  return (
    <Modal title="Move Units" onClose={onClose}>
      <div className={styles.content}>
        <p className={styles.info}>
          Moving from tile to adjacent tile.
        </p>
        <div className={styles.inputRow}>
          <label className={styles.label}>Units to move (max {maxUnits})</label>
          <input
            type="number"
            className={styles.input}
            value={units}
            min={1}
            max={maxUnits}
            onChange={e => setUnits(Math.min(maxUnits, Math.max(1, parseInt(e.target.value) || 1)))}
            autoFocus
          />
        </div>
        <label className={styles.checkboxRow}>
          <input
            type="checkbox"
            checked={standing}
            onChange={e => setStanding(e.target.checked)}
          />
          <span>Repeat each turn</span>
        </label>
        <div className={styles.actions}>
          <Button onClick={handleConfirm} variant="primary" disabled={units < 1 || units > maxUnits}>
            Confirm
          </Button>
          {existingOrder !== null && (
            <Button onClick={onCancel} variant="danger">
              Cancel Order
            </Button>
          )}
          <Button onClick={onClose} variant="secondary">
            Close
          </Button>
        </div>
      </div>
    </Modal>
  )
}
