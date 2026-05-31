type Accent = 'green' | 'blue' | 'amber' | 'teal' | 'none'

const ACCENT_COLORS: Record<Accent, { text: string; border: string }> = {
  green: { text: 'var(--accent-green)',  border: 'var(--accent-green-dim)' },
  blue:  { text: 'var(--accent-blue)',   border: 'var(--accent-blue-dim)' },
  amber: { text: 'var(--accent-amber)',  border: 'var(--accent-amber-dim)' },
  teal:  { text: 'var(--accent-teal)',   border: 'var(--border)' },
  none:  { text: 'var(--text-primary)',  border: 'var(--border)' },
}

interface MetricCardProps {
  label: string
  value: string
  sub: string
  accent: Accent
}

export default function MetricCard({ label, value, sub, accent }: MetricCardProps) {
  const colors = ACCENT_COLORS[accent]
  return (
    <div style={{
      background: 'var(--bg-surface)',
      border: '1px solid var(--border)',
      borderTop: `2px solid ${colors.border}`,
      borderRadius: 6,
      padding: '14px 16px',
    }}>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 10,
        color: 'var(--text-muted)',
        letterSpacing: '0.1em',
        marginBottom: 8,
        textTransform: 'uppercase',
      }}>
        {label}
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 22,
        fontWeight: 600,
        color: colors.text,
        lineHeight: 1,
        marginBottom: 6,
      }}>
        {value}
      </div>
      <div style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--text-muted)',
      }}>
        {sub}
      </div>
    </div>
  )
}
