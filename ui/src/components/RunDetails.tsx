import { LeaderboardEntry } from '@/types'

interface Props { entry: LeaderboardEntry }

function Row({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
        {label}
      </span>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: color || 'var(--text-primary)', fontWeight: 500 }}>
        {value}
      </span>
    </div>
  )
}

export default function RunDetails({ entry }: Props) {
  const ciWidth = ((entry.accuracy_ci_upper - entry.accuracy_ci_lower) * 100).toFixed(1)

  return (
    <div style={{
      background: 'var(--bg-surface)', border: '1px solid var(--border)',
      borderRadius: 8, padding: '1.25rem',
      height: '100%',
    }}>
      <div style={{
        fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)',
        letterSpacing: '0.08em', marginBottom: 12,
      }}>
        {entry.run_id.substring(0, 8).toUpperCase()}...
      </div>

      <div style={{ marginBottom: 16 }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 600, color: 'var(--text-primary)' }}>
          {entry.model_display_name}
        </div>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
          {entry.domain} domain
        </div>
      </div>

      <Row label="ACCURACY" value={`${(entry.accuracy_mean * 100).toFixed(1)}%`} color="var(--accent-green)" />
      <Row
        label="95% CI"
        value={`[${(entry.accuracy_ci_lower * 100).toFixed(1)}%, ${(entry.accuracy_ci_upper * 100).toFixed(1)}%]`}
        color="var(--text-secondary)"
      />
      <Row label="CI WIDTH" value={`${ciWidth}pp`} color={parseFloat(ciWidth) < 5 ? 'var(--accent-green)' : 'var(--accent-amber)'} />
      <Row
        label="HALLUCINATION RATE"
        value={`${(entry.hallucination_rate_mean * 100).toFixed(1)}%`}
        color={entry.hallucination_rate_mean > 0.3 ? 'var(--accent-amber)' : 'var(--accent-green)'}
      />
      <Row label="GROUNDING SCORE" value={`${(entry.grounding_score_mean * 100).toFixed(0)}%`} color="var(--accent-teal)" />
      <Row label="AVG COST/CALL" value={`$${entry.avg_cost_usd.toFixed(5)}`} />
      <Row label="COST/1K CALLS" value={`$${(entry.avg_cost_usd * 1000).toFixed(2)}`} />
      <Row label="P95 LATENCY" value={entry.p95_latency_ms > 0 ? `${(entry.p95_latency_ms / 1000).toFixed(2)}s` : '—'} />
      <Row
        label="CI GATE"
        value={entry.ci_gate_passed ? '✓ PASSED' : '✗ FAILED'}
        color={entry.ci_gate_passed ? 'var(--accent-green)' : 'var(--accent-red)'}
      />

      {/* Interpretation note */}
      <div style={{
        marginTop: 16, padding: '10px 12px',
        background: 'var(--bg-elevated)', borderRadius: 4,
        border: '1px solid var(--border)',
      }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)', letterSpacing: '0.06em', marginBottom: 4 }}>
          INTERPRETATION
        </div>
        <p style={{ fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.6 }}>
          A CI width of {ciWidth}pp means accuracy is measured with {parseFloat(ciWidth) < 5 ? 'high' : 'moderate'} statistical
          precision. {entry.ci_gate_passed
            ? 'This run passed all CI gates — safe to use as a baseline for regression testing.'
            : 'This run failed CI gates — review hallucination rate and accuracy thresholds before deploying.'
          }
        </p>
      </div>
    </div>
  )
}
