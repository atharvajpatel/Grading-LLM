import { Canvas } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'
import { useMemo } from 'react'
import { SCALE_COLORS, SCALE_ORDER } from '../api/client'

interface Props {
  coords: number[][]
  scaleLabels: string[]
  explainedVariance: number[]
}

// Scale factor to make -1 to 1 cube visible in 3D space
// -1 maps to -SCENE_SCALE, 0 maps to 0, 1 maps to SCENE_SCALE
const SCENE_SCALE = 1.5

// Normalize PCA coords to -1 to 1 range per dimension
function normalizeToRange(coords: number[][]): { normalized: number[][]; mins: number[]; maxs: number[] } {
  if (coords.length === 0) return { normalized: [], mins: [0, 0, 0], maxs: [1, 1, 1] }

  const dims = coords[0].length
  const mins: number[] = []
  const maxs: number[] = []

  // Find min/max for each dimension
  for (let d = 0; d < dims; d++) {
    const values = coords.map(c => c[d])
    mins.push(Math.min(...values))
    maxs.push(Math.max(...values))
  }

  // Normalize each coordinate to -1 to 1
  const normalized = coords.map(coord =>
    coord.map((v, d) => {
      const range = maxs[d] - mins[d]
      // Map to -1 to 1: first normalize to 0-1, then scale to -1 to 1
      const norm01 = range > 0 ? (v - mins[d]) / range : 0.5
      return norm01 * 2 - 1  // Convert 0-1 to -1 to 1
    })
  )

  return { normalized, mins, maxs }
}

function Points({ coords, scaleLabels }: { coords: number[][]; scaleLabels: string[] }) {
  // Normalize to -1 to 1 cube
  const { normalized: allNormalized } = useMemo(() => normalizeToRange(coords), [coords])

  // Group normalized coords by scale (preserving order)
  const normalizedByScale = useMemo(() => {
    const grouped: Record<string, number[][]> = {}
    coords.forEach((_, i) => {
      const scale = scaleLabels[i]
      if (!grouped[scale]) grouped[scale] = []
      grouped[scale].push(allNormalized[i])
    })
    return grouped
  }, [coords, scaleLabels, allNormalized])

  // Increased jitter to separate overlapping points (in 0-1 space)
  const jitterAmount = 0.06

  return (
    <>
      {SCALE_ORDER.map((scale) => {
        const scaleCoords = normalizedByScale[scale] || []
        const color = SCALE_COLORS[scale]

        return scaleCoords.map((coord, i) => {
          // Scale to scene space and add jitter
          const position = coord.map((v) => {
            const jitter = (Math.random() - 0.5) * jitterAmount
            return (v + jitter) * SCENE_SCALE
          })
          return (
            <mesh key={`${scale}-${i}`} position={position as [number, number, number]}>
              <sphereGeometry args={[0.04, 16, 16]} />
              <meshStandardMaterial
                color={color}
                emissive={color}
                emissiveIntensity={0.4}
                opacity={0.95}
                transparent
              />
            </mesh>
          )
        })
      })}
    </>
  )
}

function Centroids({ coords, scaleLabels }: { coords: number[][]; scaleLabels: string[] }) {
  const { normalized: allNormalized } = useMemo(() => normalizeToRange(coords), [coords])

  // Compute centroid (mean position) and spread for each scale
  const centroidData = useMemo(() => {
    const result: Record<string, { centroid: number[]; spread: number; uniquePoints: number }> = {}

    SCALE_ORDER.forEach(scale => {
      const indices = scaleLabels.map((s, i) => s === scale ? i : -1).filter(i => i >= 0)
      if (indices.length > 0) {
        const centroid = [0, 1, 2].map(dim =>
          indices.reduce((sum, i) => sum + allNormalized[i][dim], 0) / indices.length
        )

        // Calculate spread (average distance from centroid)
        const distances = indices.map(i => {
          const point = allNormalized[i]
          return Math.sqrt(
            Math.pow(point[0] - centroid[0], 2) +
            Math.pow(point[1] - centroid[1], 2) +
            Math.pow(point[2] - centroid[2], 2)
          )
        })
        const spread = distances.reduce((sum, d) => sum + d, 0) / distances.length

        // Count unique points (within small tolerance)
        const uniquePoints = new Set(
          indices.map(i => allNormalized[i].map(v => Math.round(v * 100) / 100).join(','))
        ).size

        result[scale] = { centroid, spread, uniquePoints }
      }
    })

    return result
  }, [scaleLabels, allNormalized])

  return (
    <>
      {SCALE_ORDER.map(scale => {
        const data = centroidData[scale]
        if (!data) return null

        // Hide centroid if points are too clustered (spread < 0.05) or only 1-2 unique points
        if (data.spread < 0.05 || data.uniquePoints <= 2) return null

        const position = data.centroid.map(v => v * SCENE_SCALE)
        const color = SCALE_COLORS[scale]

        return (
          <mesh key={`centroid-${scale}`} position={position as [number, number, number]}>
            <octahedronGeometry args={[0.12]} />
            <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} />
          </mesh>
        )
      })}
    </>
  )
}

function Grid() {
  // Grid for -1 to 1 cube (scaled to scene space)
  const gridLines: Array<{ start: [number, number, number]; end: [number, number, number] }> = []
  const tickValues = [-1, -0.5, 0, 0.5, 1]

  // XY plane (z=-1) - back wall
  tickValues.forEach(x => {
    // Vertical lines (along Y)
    gridLines.push({
      start: [x * SCENE_SCALE, -SCENE_SCALE, -SCENE_SCALE],
      end: [x * SCENE_SCALE, SCENE_SCALE, -SCENE_SCALE]
    })
  })
  tickValues.forEach(y => {
    // Horizontal lines (along X)
    gridLines.push({
      start: [-SCENE_SCALE, y * SCENE_SCALE, -SCENE_SCALE],
      end: [SCENE_SCALE, y * SCENE_SCALE, -SCENE_SCALE]
    })
  })

  // XZ plane (y=-1) - floor grid
  tickValues.forEach(x => {
    gridLines.push({
      start: [x * SCENE_SCALE, -SCENE_SCALE, -SCENE_SCALE],
      end: [x * SCENE_SCALE, -SCENE_SCALE, SCENE_SCALE]
    })
  })
  tickValues.forEach(z => {
    gridLines.push({
      start: [-SCENE_SCALE, -SCENE_SCALE, z * SCENE_SCALE],
      end: [SCENE_SCALE, -SCENE_SCALE, z * SCENE_SCALE]
    })
  })

  // YZ plane (x=-1) - side wall
  tickValues.forEach(y => {
    gridLines.push({
      start: [-SCENE_SCALE, y * SCENE_SCALE, -SCENE_SCALE],
      end: [-SCENE_SCALE, y * SCENE_SCALE, SCENE_SCALE]
    })
  })
  tickValues.forEach(z => {
    gridLines.push({
      start: [-SCENE_SCALE, -SCENE_SCALE, z * SCENE_SCALE],
      end: [-SCENE_SCALE, SCENE_SCALE, z * SCENE_SCALE]
    })
  })

  return (
    <>
      {gridLines.map((line, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([...line.start, ...line.end])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#ccc" transparent opacity={0.3} />
        </line>
      ))}
    </>
  )
}

function AxisTicks({ explainedVariance: _explainedVariance }: { explainedVariance: number[] }) {
  const tickValues = [-1, -0.5, 0, 0.5, 1]
  const displayTicks = [-1, 0, 1] // Only show labels for -1, 0, 1 to reduce clutter

  return (
    <>
      {/* X axis ticks (PC1) */}
      {tickValues.map(v => (
        <group key={`x-${v}`}>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([v * SCENE_SCALE, -0.05, 0, v * SCENE_SCALE, 0.05, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#aaa" />
          </line>
          {displayTicks.includes(v) && (
            <Text position={[v * SCENE_SCALE, -0.2, 0]} fontSize={0.12} color="#666">
              {v}
            </Text>
          )}
        </group>
      ))}

      {/* Y axis ticks (PC2) */}
      {tickValues.map(v => (
        <group key={`y-${v}`}>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([-0.05, v * SCENE_SCALE, 0, 0.05, v * SCENE_SCALE, 0])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#aaa" />
          </line>
          {displayTicks.includes(v) && v !== 0 && (
            <Text position={[-0.2, v * SCENE_SCALE, 0]} fontSize={0.12} color="#666">
              {v}
            </Text>
          )}
        </group>
      ))}

      {/* Z axis ticks (PC3) */}
      {tickValues.map(v => (
        <group key={`z-${v}`}>
          <line>
            <bufferGeometry>
              <bufferAttribute
                attach="attributes-position"
                count={2}
                array={new Float32Array([0, -0.05, v * SCENE_SCALE, 0, 0.05, v * SCENE_SCALE])}
                itemSize={3}
              />
            </bufferGeometry>
            <lineBasicMaterial color="#aaa" />
          </line>
          {displayTicks.includes(v) && v !== 0 && (
            <Text position={[0, -0.2, v * SCENE_SCALE]} fontSize={0.12} color="#666">
              {v}
            </Text>
          )}
        </group>
      ))}
    </>
  )
}

function Axes({ explainedVariance }: { explainedVariance: number[] }) {
  return (
    <>
      {/* X axis - PC1 (-1 to 1) */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([-SCENE_SCALE, 0, 0, SCENE_SCALE, 0, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#666" />
      </line>
      <Text
        position={[SCENE_SCALE + 0.3, 0, 0]}
        fontSize={0.15}
        color="#444"
      >
        PC1 ({(explainedVariance[0] * 100).toFixed(0)}%)
      </Text>

      {/* Y axis - PC2 (-1 to 1) */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, -SCENE_SCALE, 0, 0, SCENE_SCALE, 0])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#666" />
      </line>
      <Text
        position={[0, SCENE_SCALE + 0.3, 0]}
        fontSize={0.15}
        color="#444"
      >
        PC2 ({(explainedVariance[1] * 100).toFixed(0)}%)
      </Text>

      {/* Z axis - PC3 (-1 to 1) */}
      <line>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={2}
            array={new Float32Array([0, 0, -SCENE_SCALE, 0, 0, SCENE_SCALE])}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#666" />
      </line>
      <Text
        position={[0, 0, SCENE_SCALE + 0.3]}
        fontSize={0.15}
        color="#444"
      >
        PC3 ({(explainedVariance[2] * 100).toFixed(0)}%)
      </Text>

      {/* Origin marker */}
      <mesh position={[0, 0, 0]}>
        <sphereGeometry args={[0.04, 8, 8]} />
        <meshStandardMaterial color="#999" />
      </mesh>
    </>
  )
}

export default function PCAPlot3D({ coords, scaleLabels, explainedVariance }: Props) {
  return (
    <Canvas camera={{ position: [4, 4, 4], fov: 50 }}>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} />
      <Grid />
      <AxisTicks explainedVariance={explainedVariance} />
      <Points coords={coords} scaleLabels={scaleLabels} />
      <Centroids coords={coords} scaleLabels={scaleLabels} />
      <Axes explainedVariance={explainedVariance} />
      <OrbitControls enablePan enableZoom enableRotate target={[0, 0, 0]} />
    </Canvas>
  )
}
