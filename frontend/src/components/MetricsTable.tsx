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
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200">
            <th className="text-left py-3 px-4 font-medium text-gray-600">Scale</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">Avg Variance</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">Avg Consistency</th>
            <th className="text-right py-3 px-4 font-medium text-gray-600">Avg Entropy</th>
          </tr>
        </thead>
        <tbody>
          {SCALE_ORDER.map((scale) => {
            const metrics = scaleMetrics[scale]
            if (!metrics) return null

            return (
              <tr key={scale} className="border-b border-gray-100 hover:bg-gray-50">
                <td className="py-3 px-4">
                  <div className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: SCALE_COLORS[scale] }}
                    />
                    <span className="font-medium capitalize">{scale}</span>
                  </div>
                </td>
                <td className="text-right py-3 px-4 font-mono">
                  {metrics.avg_variance.toFixed(4)}
                </td>
                <td className="text-right py-3 px-4 font-mono">
                  {(metrics.avg_consistency * 100).toFixed(1)}%
                </td>
                <td className="text-right py-3 px-4 font-mono">
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
