import { SCALE_ORDER, SCALE_COLORS } from '../api/client'

interface Props {
  embeddings: Record<string, number[][]> // scale -> n_samples x n_questions array
}

// Per-scale legends showing valid values and their meanings
const SCALE_LEGENDS: Record<string, Array<{ value: number; label: string }>> = {
  binary: [
    { value: 0, label: 'Absent' },
    { value: 1, label: 'Present' },
  ],
  ternary: [
    { value: 0, label: 'Absent' },
    { value: 0.5, label: 'Peripheral' },
    { value: 1, label: 'Central' },
  ],
  quaternary: [
    { value: 0, label: 'Absent' },
    { value: 0.33, label: 'Weak' },
    { value: 0.66, label: 'Moderate' },
    { value: 1, label: 'Strong' },
  ],
}

// Generate CSS gradient for continuous scale
function getContinuousGradient(): string {
  return 'linear-gradient(to right, rgb(220, 0, 60), rgb(220, 220, 60), rgb(0, 220, 60))'
}

// Valid values for snapping (discrete scales)
const SCALE_VALUES: Record<string, number[]> = {
  binary: [0, 1],
  ternary: [0, 0.5, 1],
  quaternary: [0, 0.33, 0.66, 1],
}

// Red (0) → Yellow (0.5) → Green (1) gradient for ALL scales
function scoreToColor(score: number): string {
  let r: number, g: number
  if (score <= 0.5) {
    // Red to Yellow: keep red high, increase green
    r = 220
    g = Math.round(score * 2 * 220)
  } else {
    // Yellow to Green: decrease red, keep green high
    r = Math.round(220 * (1 - (score - 0.5) * 2))
    g = 220
  }
  return `rgb(${r}, ${g}, 60)`
}

// Compute distribution of values for a scale
function computeDistribution(vectors: number[][], scale: string): Array<{ value: number; label: string; count: number; percent: number }> {
  const allValues = vectors.flat()
  const total = allValues.length

  if (scale === 'continuous') {
    // For continuous, bucket into ranges
    const buckets = [
      { value: 0, label: '0', min: -0.05, max: 0.15 },
      { value: 0.25, label: '0.1-0.4', min: 0.15, max: 0.45 },
      { value: 0.5, label: '0.5', min: 0.45, max: 0.55 },
      { value: 0.75, label: '0.6-0.9', min: 0.55, max: 0.95 },
      { value: 1, label: '1', min: 0.95, max: 1.05 },
    ]
    return buckets.map(bucket => {
      const count = allValues.filter(v => v >= bucket.min && v < bucket.max).length
      return { value: bucket.value, label: bucket.label, count, percent: (count / total) * 100 }
    })
  }

  // For discrete scales, snap to nearest valid value and count
  const validValues = SCALE_VALUES[scale] || [0, 1]
  const snapToNearest = (v: number) => validValues.reduce((prev, curr) =>
    Math.abs(curr - v) < Math.abs(prev - v) ? curr : prev
  )

  return SCALE_LEGENDS[scale].map(({ value, label }) => {
    const count = allValues.filter(v => Math.abs(snapToNearest(v) - value) < 0.01).length
    return { value, label, count, percent: (count / total) * 100 }
  })
}

// Compute consistency (% matching mode) and entropy
function computeStats(vectors: number[][], scale: string): { consistency: number; entropy: number } {
  const allValues = vectors.flat()
  const total = allValues.length

  // Snap values to nearest valid value for discrete scales
  const validValues = scale === 'continuous' ? null : SCALE_VALUES[scale]
  const snapped = validValues
    ? allValues.map(v => validValues.reduce((prev, curr) => Math.abs(curr - v) < Math.abs(prev - v) ? curr : prev))
    : allValues.map(v => Math.round(v * 10) / 10) // Round continuous to 0.1

  // Count occurrences
  const counts: Record<string, number> = {}
  snapped.forEach(v => {
    const key = v.toFixed(2)
    counts[key] = (counts[key] || 0) + 1
  })

  // Mode consistency = max count / total
  const maxCount = Math.max(...Object.values(counts))
  const consistency = maxCount / total

  // Shannon entropy
  const probs = Object.values(counts).map(c => c / total).filter(p => p > 0)
  const entropy = -probs.reduce((sum, p) => sum + p * Math.log2(p), 0)

  return { consistency, entropy }
}

export default function ValueDistribution({ embeddings }: Props) {
  if (!embeddings || Object.keys(embeddings).length === 0) {
    return <div className="text-gray-500">No distribution data available</div>
  }

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-gray-800">Value Distribution by Scale</h3>

      {/* Metric Definitions */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h4 className="font-semibold text-gray-800 mb-3">Metric Definitions</h4>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="font-medium text-gray-700">Consistency (% matching mode)</p>
            <p className="text-gray-600 mb-1">
              Percentage of all values that match the most common value.
            </p>
            <p className="text-green-600 text-xs">Higher = more agreement (values cluster at one option)</p>
            <p className="text-red-600 text-xs">Lower = more disagreement (values spread across options)</p>
          </div>
          <div>
            <p className="font-medium text-gray-700">Entropy (Shannon entropy, bits)</p>
            <p className="text-gray-600 mb-1">
              Measures the spread/unpredictability of values.
            </p>
            <p className="text-green-600 text-xs">Lower = concentrated, predictable (e.g., 0.0 = all same)</p>
            <p className="text-red-600 text-xs">Higher = spread out, unpredictable (max depends on # options)</p>
            <div className="mt-2 text-xs text-gray-500 bg-white p-2 rounded border">
              <p className="font-medium text-gray-600 mb-1">Entropy Scale:</p>
              <p><span className="text-green-700">0.0</span> = perfect agreement (all identical)</p>
              <p><span className="text-yellow-600">0.5-1.0</span> = slight spread, one option dominates</p>
              <p><span className="text-orange-600">1.0-1.5</span> = moderate spread across options</p>
              <p><span className="text-red-600">1.5-2.0+</span> = high disagreement, values scattered</p>
            </div>
          </div>
        </div>
      </div>

      {/* Distribution Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {SCALE_ORDER.map((scale) => {
          const vectors = embeddings[scale]
          if (!vectors || vectors.length === 0) return null
          const distribution = computeDistribution(vectors, scale)
          const stats = computeStats(vectors, scale)

          return (
            <div
              key={scale}
              className="p-4 bg-white rounded-lg border-2"
              style={{ borderColor: SCALE_COLORS[scale] }}
            >
              <h4
                className="font-semibold capitalize mb-3 text-center text-lg"
                style={{ color: SCALE_COLORS[scale] }}
              >
                {scale}
              </h4>

              {/* Color legend - gradient for continuous, discrete for others */}
              {scale === 'continuous' ? (
                <div className="flex items-center justify-center gap-2 mb-3">
                  <span className="text-xs text-gray-600">0</span>
                  <div
                    className="w-32 h-4 rounded border border-gray-300"
                    style={{ background: getContinuousGradient() }}
                  />
                  <span className="text-xs text-gray-600">1</span>
                </div>
              ) : null}

              {/* Distribution bars */}
              <div className="mb-4 px-2">
                <div className="flex items-end gap-2 h-24">
                  {distribution.map(({ value, label, percent }) => (
                    <div key={value} className="flex-1 flex flex-col items-center">
                      <div
                        className="w-full rounded-t"
                        style={{
                          height: `${Math.max(percent * 0.8, 4)}px`,
                          backgroundColor: scoreToColor(value),
                          border: '1px solid rgba(0,0,0,0.15)',
                        }}
                      />
                      <span className="text-sm font-medium text-gray-700 mt-1">{percent.toFixed(0)}%</span>
                      <span className="text-xs text-gray-500">{label}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Stats row */}
              <div className="flex justify-center gap-6 text-sm border-t border-gray-200 pt-3">
                <span className="text-gray-600">
                  Consistency: <strong className="text-gray-800">{(stats.consistency * 100).toFixed(0)}%</strong>
                </span>
                <span className="text-gray-600">
                  Entropy: <strong className="text-gray-800">{stats.entropy.toFixed(2)}</strong>
                </span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
