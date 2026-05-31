'use client'

import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell, ReferenceLine,
} from 'recharts'
import { ParetoPoint } from '@/types'

interface Props { data: ParetoPoint[] }

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload as ParetoPoint
  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border)',
      borderRadius: 6, padding: '10px 14px',
      fontFamily: 'var(--font-mono)', fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-primary)', fontWeight: 600, marginBottom: 6 }}>{d.model}</div>
      <div style={{ color: 'var(--text-secondary)' }}>Quality: <span style={{ color: 'var(--accent-green)' }}>{d.quality.toFixed(1)}%</span></div>
      <div style={{ color: 'var(--text-secondary)' }}>Cost/1K: <span style={{ color: 'var(--accent-amber)' }}>${d.cost.toFixed(2)}</span></div>
      <div style={{ color: 'var(--text-secondary)' }}>P95 latency: <span style={{ color: 'var(--accent-blue)' }}>{(d.latency / 1000).toFixed(1)}s</span></div>
      {d.isPareto && (
        <div style={{ color: 'var(--accent-teal)', marginTop: 4, fontSize: 10, letterSpacing: '0.06em' }}>
          ★ PARETO OPTIMAL
        </div>
      )}
    </div>
  )
}

const CustomDot = (props: any) => {
  const { cx, cy, payload } = props
  const isPareto = payload.isPareto
  return (
    <g>
      {isPareto && (
        <circle cx={cx} cy={cy} r={14} fill="var(--accent-green)" fillOpacity={0.08} />
      )}
      <circle
        cx={cx} cy={cy}
        r={isPareto ? 7 : 5}
        fill={isPareto ? 'var(--accent-green)' : 'var(--accent-blue)'}
        stroke={isPareto ? 'var(--accent-green)' : 'var(--accent-blue)'}
        strokeWidth={isPareto ? 2 : 1}
        fillOpacity={isPareto ? 0.9 : 0.6}
      />
      <text
        x={cx} y={cy - 12}
        textAnchor="middle"
        fill="var(--text-secondary)"
        fontSize={10}
        fontFamily="IBM Plex Mono, monospace"
      >
        {payload.model.replace('GPT-', 'GPT-').replace('Claude ', 'C-')}
      </text>
    </g>
  )
}

export default function ParetoChart({ data }: Props) {
  if (data.length === 0) {
    return (
      <div style={{
        height: 280, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 12,
      }}>
        Run multiple models to see the Pareto frontier
      </div>
    )
  }

  // If only one point, show a placeholder message with the point
  if (data.length === 1) {
    const d = data[0]
    return (
      <div style={{ height: 280, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: 12 }}>
        <div style={{
          border: '1px solid var(--accent-green)',
          borderRadius: 8, padding: '16px 24px', textAlign: 'center',
        }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-teal)', letterSpacing: '0.08em', marginBottom: 8 }}>
            ★ CURRENT BASELINE — PARETO OPTIMAL
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 20, fontWeight: 600, color: 'var(--accent-green)' }}>
            {d.model}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', marginTop: 8 }}>
            {d.quality.toFixed(1)}% accuracy · ${d.cost.toFixed(2)}/1K calls
          </div>
        </div>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', textAlign: 'center' }}>
          Run claude-sonnet-4-6 or gemini-1.5-pro to populate the frontier
        </p>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={280}>
      <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
        <XAxis
          dataKey="cost" name="Cost/1K calls"
          tick={{ fontFamily: 'IBM Plex Mono', fontSize: 10, fill: 'var(--text-muted)' }}
          label={{ value: 'Cost/1K calls (USD)', position: 'insideBottom', offset: -10, fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
        />
        <YAxis
          dataKey="quality" name="Accuracy"
          domain={[80, 100]}
          tick={{ fontFamily: 'IBM Plex Mono', fontSize: 10, fill: 'var(--text-muted)' }}
          label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 10, fontFamily: 'IBM Plex Mono' }}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3', stroke: 'var(--border)' }} />
        <Scatter data={data} shape={<CustomDot />} />
      </ScatterChart>
    </ResponsiveContainer>
  )
}
