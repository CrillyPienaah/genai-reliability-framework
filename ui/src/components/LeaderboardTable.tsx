import { LeaderboardEntry } from '@/types'

interface Props {
  entries: LeaderboardEntry[]
  selectedId: string | null
  onSelect: (id: string) => void
}

function CIBar({ mean, low, high, color }: { mean: number; low: number; high: number; color: string }) {
  // Render a bar with CI whiskers scaled 0–100
  const barLeft = `${(low * 100).toFixed(1)}%`
  const barWidth = `${((high - low) * 100).toFixed(1)}%`
  const dotLeft = `${(mean * 100).toFixed(1)}%`

  return (
    <div style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center' }}>
      {/* Track */}
      <div style={{
        position: 'absolute', left: 0, right: 0, height: 3,
        background: 'var(--bg-highlight)', borderRadius: 2,
      }} />
      {/* CI range */}
      <div style={{
        position: 'absolute', left: barLeft, width: barWidth, height: 3,
        background: color, opacity: 0.35, borderRadius: 2,
      }} />
      {/* Mean dot */}
      <div style={{
        position: 'absolute', left: dotLeft, width: 7, height: 7,
        borderRadius: '50%', background: color,
        transform: 'translateX(-50%)',
        boxShadow: `0 0 4px ${color}`,
      }} />
    </div>
  )
}

const STATUS_BADGE = (passed: boolean) => (
  <span style={{
    display: 'inline-flex', alignItems: 'center', gap: 4,
    padding: '2px 8px', borderRadius: 3,
    background: passed ? 'var(--accent-green-dim)' : 'var(--accent-red-dim)',
    color: passed ? 'var(--accent-green)' : 'var(--accent-red)',
    fontFamily: 'var(--font-mono)', fontSize: 10, letterSpacing: '0.06em',
    fontWeight: 500,
  }}>
    {passed ? '✓ PASS' : '✗ FAIL'}
  </span>
)

export default function LeaderboardTable({ entries, selectedId, onSelect }: Props) {
  if (entries.length === 0) {
    return (
      <div style={{
        border: '1px solid var(--border)', borderRadius: 8,
        padding: '3rem', textAlign: 'center',
        color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 12,
      }}>
        No runs yet for this domain. Run: python -m src.cli run --model gpt-4o-mini --domain medical --n 30
      </div>
    )
  }

  const cols = [
    { key: 'model', label: 'MODEL', width: '160px' },
    { key: 'accuracy', label: 'ACCURACY + 95% CI', width: '220px' },
    { key: 'hallucination', label: 'HALLUCINATION RATE', width: '160px' },
    { key: 'grounding', label: 'GROUNDING', width: '120px' },
    { key: 'cost', label: 'COST/CALL', width: '100px' },
    { key: 'latency', label: 'P95 LATENCY', width: '110px' },
    { key: 'gate', label: 'CI GATE', width: '90px' },
  ]

  return (
    <div style={{
      border: '1px solid var(--border)', borderRadius: 8,
      overflow: 'hidden', background: 'var(--bg-surface)',
    }}>
      {/* Header */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: cols.map(c => c.width).join(' '),
        padding: '10px 16px',
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-elevated)',
      }}>
        {cols.map(c => (
          <div key={c.key} style={{
            fontFamily: 'var(--font-mono)', fontSize: 10,
            color: 'var(--text-muted)', letterSpacing: '0.1em',
          }}>
            {c.label}
          </div>
        ))}
      </div>

      {/* Rows */}
      {entries.map((entry, i) => {
        const isSelected = entry.run_id === selectedId
        return (
          <div
            key={entry.run_id}
            onClick={() => onSelect(entry.run_id)}
            style={{
              display: 'grid',
              gridTemplateColumns: cols.map(c => c.width).join(' '),
              padding: '14px 16px',
              borderBottom: i < entries.length - 1 ? '1px solid var(--border)' : 'none',
              cursor: 'pointer',
              background: isSelected ? 'var(--bg-highlight)' : 'transparent',
              transition: 'background 0.15s',
              alignItems: 'center',
            }}
            onMouseEnter={e => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-elevated)' }}
            onMouseLeave={e => { if (!isSelected) (e.currentTarget as HTMLDivElement).style.background = 'transparent' }}
          >
            {/* Model */}
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>
                {entry.model_display_name}
              </div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', marginTop: 2 }}>
                {entry.domain}
              </div>
            </div>

            {/* Accuracy + CI bar */}
            <div style={{ paddingRight: 16 }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 600, color: 'var(--accent-green)', marginBottom: 4 }}>
                {(entry.accuracy_mean * 100).toFixed(1)}%
                <span style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 6 }}>
                  [{(entry.accuracy_ci_lower * 100).toFixed(1)}–{(entry.accuracy_ci_upper * 100).toFixed(1)}]
                </span>
              </div>
              <CIBar mean={entry.accuracy_mean} low={entry.accuracy_ci_lower} high={entry.accuracy_ci_upper} color="var(--accent-green)" />
            </div>

            {/* Hallucination rate */}
            <div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: entry.hallucination_rate_mean > 0.3 ? 'var(--accent-amber)' : 'var(--accent-green)' }}>
                {(entry.hallucination_rate_mean * 100).toFixed(1)}%
              </div>
            </div>

            {/* Grounding */}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent-teal)' }}>
              {(entry.grounding_score_mean * 100).toFixed(0)}%
            </div>

            {/* Cost */}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)' }}>
              ${entry.avg_cost_usd.toFixed(4)}
            </div>

            {/* Latency */}
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)' }}>
              {entry.p95_latency_ms > 0 ? `${(entry.p95_latency_ms / 1000).toFixed(1)}s` : '—'}
            </div>

            {/* CI gate */}
            <div>{STATUS_BADGE(entry.ci_gate_passed)}</div>
          </div>
        )
      })}
    </div>
  )
}
