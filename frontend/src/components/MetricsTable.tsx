import { SCALE_ORDER, SCALE_COLORS } from '../api/client'

interface ScaleMetrics {
  avg_variance: number
  avg_consistency: number
  avg_entropy: number
}

interface Props {
  scaleMetrics: Record<string, ScaleMetrics>
}

export default function MetricsTable({ scaleMetrics }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="data-table w-full">
        <thead>
          <tr>
            <th className="text-left">Scale</th>
            <th className="text-right">Avg Variance</th>
            <th className="text-right">Avg Consistency</th>
            <th className="text-right">Avg Entropy</th>
          </tr>
        </thead>
        <tbody>
          {SCALE_ORDER.map((scale) => {
            const metrics = scaleMetrics[scale]
            if (!metrics) return null

            return (
              <tr key={scale}>
                <td>
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full border border-hair"
                      style={{ backgroundColor: SCALE_COLORS[scale] }}
                    />
                    <span className="font-medium capitalize text-ink">{scale}</span>
                  </div>
                </td>
                <td className="num">
                  {metrics.avg_variance.toFixed(4)}
                </td>
                <td className="num">
                  {(metrics.avg_consistency * 100).toFixed(1)}%
                </td>
                <td className="num">
                  {metrics.avg_entropy.toFixed(3)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
