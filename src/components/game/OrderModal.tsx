import { useState, useEffect } from 'react'
import { Modal } from '../shared/Modal'
import { Button } from '../shared/Button'
import { UNITS_ALL } from '../../types/orders'
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
  const isExistingAll = existingOrder === UNITS_ALL
  const [units, setUnits] = useState(Math.min(Math.max(1, isExistingAll ? maxUnits : (existingOrder ?? maxUnits)), maxUnits))
  const [standing, setStanding] = useState(isStanding)
  const [allUnits, setAllUnits] = useState(isExistingAll)

  useEffect(() => {
    const existingAll = existingOrder === UNITS_ALL
    setUnits(Math.min(Math.max(1, existingAll ? maxUnits : (existingOrder ?? maxUnits)), maxUnits))
    setStanding(isStanding)
    setAllUnits(existingAll)
  }, [existingOrder, maxUnits, isStanding])

  const adjust = (delta: number) =>
    setUnits(u => Math.min(maxUnits, Math.max(1, u + delta)))

  const handleConfirm = () => {
    const finalUnits = standing && allUnits ? UNITS_ALL : units
    if (finalUnits === UNITS_ALL || (finalUnits >= 1 && finalUnits <= maxUnits)) {
      onConfirm(finalUnits, standing)
    }
  }

  const stepperDisabled = standing && allUnits

  return (
    <Modal title="Move Units" onClose={onClose} maxWidth={300}>
      <div className={styles.content}>
        <div className={styles.stepper}>
          <span className={styles.stepValue}>
            {stepperDisabled ? '∞' : units}
          </span>
          <button
            className={styles.stepBtn}
            onClick={() => adjust(-1)}
            disabled={stepperDisabled || units <= 1}
          >−</button>
          <button
            className={styles.stepBtn}
            onClick={() => adjust(1)}
            disabled={stepperDisabled || units >= maxUnits}
          >+</button>
        </div>

        <div className={styles.presets}>
          <button
            className={styles.presetBtn}
            onClick={() => setUnits(maxUnits)}
            disabled={stepperDisabled || units === maxUnits}
          >Max</button>
          {standing && (
            <button
              className={`${styles.presetBtn} ${styles.presetBtnInfinity} ${allUnits ? styles.presetBtnActive : ''}`}
              onClick={() => setAllUnits(a => !a)}
            >∞</button>
          )}
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
          <Button onClick={onClose} variant="secondary">Close</Button>
          <span style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem' }}>
            {existingOrder !== null && (
              <Button onClick={onCancel} variant="danger">Cancel</Button>
            )}
            <Button onClick={handleConfirm} variant="primary">Confirm</Button>
          </span>
        </div>
      </div>
    </Modal>
  )
}
