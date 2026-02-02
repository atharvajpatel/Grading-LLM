import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import { SCALE_ORDER, SCALE_COLORS } from '../api/client'

interface ScaleMetrics {
  avg_variance: number
  avg_consistency: number
  avg_entropy: number
}

interface Props {
  scaleMetrics: Record<string, ScaleMetrics>
}

export default function VarianceChart({ scaleMetrics }: Props) {
  const data = SCALE_ORDER.map((scale) => ({
    name: scale.charAt(0).toUpperCase() + scale.slice(1),
    variance: scaleMetrics[scale]?.avg_variance || 0,
    color: SCALE_COLORS[scale],
  }))

  return (
    <div className="h-64">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis dataKey="name" tick={{ fill: '#6b7280' }} />
          <YAxis tick={{ fill: '#6b7280' }} />
          <Tooltip
            contentStyle={{
              backgroundColor: 'white',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
            formatter={(value: number) => [value.toFixed(4), 'Variance']}
          />
          <Bar dataKey="variance" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
