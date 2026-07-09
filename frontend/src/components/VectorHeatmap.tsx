import { SCALE_ORDER, SCALE_COLORS } from '../api/client'
import { magnitudeColor, PALETTE } from '../theme/palette'

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

// Generate CSS gradient for continuous scale legend: cream -> gold -> ink,
// matching magnitudeColor (monotonic in luminance).
function getContinuousGradient(): string {
  return `linear-gradient(to right, ${PALETTE.cream}, ${PALETTE.gold}, ${PALETTE.ink})`
}

// Sequential magnitude ramp (cream -> gold -> ink) for ALL scales.
// score is already normalized to [0,1]; normalization is untouched.
function scoreToColor(score: number): string {
  return magnitudeColor(score)
}

export default function VectorHeatmap({ embeddings }: Props) {
  if (!embeddings || Object.keys(embeddings).length === 0) {
    return <div className="muted">No embedding data available</div>
  }

  return (
    <div className="space-y-6">
      <h3 className="section-title">Embedding Vectors by Scale</h3>
      <p className="text-sm text-mute">
        Each row is a sample (20 total), each column is a question (20 total).
        Cream = 0 (absent), Ink = 1 (present). Hover for exact values.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {SCALE_ORDER.map((scale) => {
          const vectors = embeddings[scale]
          if (!vectors || vectors.length === 0) return null
          const legend = SCALE_LEGENDS[scale] || []

          return (
            <div
              key={scale}
              className="card"
            >
              <h4 className="font-semibold capitalize mb-3 text-center text-lg text-ink flex items-center justify-center gap-2">
                <span
                  className="inline-block w-3 h-3 rounded-sm"
                  style={{ backgroundColor: SCALE_COLORS[scale], border: `1px solid ${PALETTE.hair}` }}
                />
                {scale}
              </h4>

              {/* Per-scale legend */}
              {scale === 'continuous' ? (
                // Continuous: show gradient bar
                <div className="flex items-center justify-center gap-2 mb-3">
                  <span className="text-xs text-mute">0</span>
                  <div
                    className="w-32 h-5 rounded border border-hair"
                    style={{ background: getContinuousGradient() }}
                  />
                  <span className="text-xs text-mute">1</span>
                </div>
              ) : (
                // Discrete scales: show individual boxes
                <div className="flex justify-center gap-3 mb-3 flex-wrap">
                  {legend.map(({ value, label }) => (
                    <div key={value} className="flex items-center gap-1">
                      <div
                        className="w-5 h-5 rounded border border-hair"
                        style={{ backgroundColor: scoreToColor(value) }}
                      />
                      <span className="text-xs text-mute">{label}</span>
                    </div>
                  ))}
                </div>
              )}

              <div className="flex flex-col gap-1 bg-cream p-2 rounded">
                {/* Column headers (question numbers) */}
                <div className="flex gap-1 mb-1">
                  <div className="w-6 text-xs text-mute text-right pr-1 flex-shrink-0">S\Q</div>
                  {vectors[0]?.map((_, colIdx) => (
                    <div
                      key={colIdx}
                      className="flex-1 min-w-[16px] h-4 text-center text-xs text-mute"
                    >
                      {colIdx + 1}
                    </div>
                  ))}
                </div>

                {/* Rows (samples) */}
                {vectors.map((vector, rowIdx) => (
                  <div key={rowIdx} className="flex gap-1 items-center">
                    <div className="w-6 text-xs text-mute text-right pr-1 flex-shrink-0">
                      {rowIdx + 1}
                    </div>
                    {vector.map((score, colIdx) => (
                      <div
                        key={colIdx}
                        className="flex-1 min-w-[16px] h-5 rounded-sm cursor-pointer hover:ring-2 hover:ring-gold"
                        style={{ backgroundColor: scoreToColor(score) }}
                        title={`Sample ${rowIdx + 1}, Q${colIdx + 1}: ${score.toFixed(3)}`}
                      />
                    ))}
                  </div>
                ))}
              </div>

              {/* Size info */}
              <div className="mt-2 text-xs text-mute text-center">
                {vectors.length} samples x {vectors[0]?.length || 0} questions
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
