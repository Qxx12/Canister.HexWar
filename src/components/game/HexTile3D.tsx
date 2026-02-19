import { useMemo } from 'react'
import * as THREE from 'three'
import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import type { Tile } from '../../types/board'
import type { PlayerId } from '../../types/player'
import { axialToPixel } from '../../types/hex'
import { PLAYER_COLORS } from '../../types/player'
import { getTerrainTexture } from './terrainTextures'

export const TILE_DEPTH = 5
export const SURFACE_Y = TILE_DEPTH / 2

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
      {tile.owner !== null && (
        <Html center position={[0, SURFACE_Y + 2, 0]} zIndexRange={[9, 0]} pointerEvents="none" style={{ pointerEvents: 'none' }}>
          <div style={{ textAlign: 'center', userSelect: 'none', whiteSpace: 'nowrap' }}>
            {tile.isStartTile && (
              <div style={{
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
