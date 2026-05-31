'use client'

import { useState, useEffect } from 'react'
import { RunSummary, LeaderboardEntry, ParetoPoint } from '@/types'
import LeaderboardTable from '@/components/LeaderboardTable'
import ParetoChart from '@/components/ParetoChart'
import MetricCard from '@/components/MetricCard'
import RunDetails from '@/components/RunDetails'

// ── Seed data: your real Week 2 run results ───────────────────────────────────
// In Week 3 final, this will be fetched from FastAPI /leaderboard
// For the demo, we seed with the actual run results from your pipeline
const SEED_RUNS: LeaderboardEntry[] = [
  {
    run_id: 'ebb12f2a-d21a-435c-8188-4a9c195bca1a',
    model_display_name: 'GPT-4o Mini',
    model_id: 'gpt-4o-mini',
    domain: 'medical',
    accuracy_mean: 0.933,
    accuracy_ci_lower: 0.908,
    accuracy_ci_upper: 0.960,
    hallucination_rate_mean: 0.500,
    grounding_score_mean: 1.000,
    avg_cost_usd: 0.0002,
    p95_latency_ms: 8175,
    ci_gate_passed: true,
  },
  {
    run_id: 'cac95fbe-827c-43c7-aae0-14aae31e5c02',
    model_display_name: 'GPT-4o Mini',
    model_id: 'gpt-4o-mini',
    domain: 'finance',
    accuracy_mean: 0.935,
    accuracy_ci_lower: 0.892,
    accuracy_ci_upper: 0.970,
    hallucination_rate_mean: 0.350,
    grounding_score_mean: 1.000,
    avg_cost_usd: 0.0002,
    p95_latency_ms: 6397,
    ci_gate_passed: true,
  },
  // Placeholder rows — populated when you run more models
  {
    run_id: 'pending-gpt4o',
    model_display_name: 'GPT-4o',
    model_id: 'gpt-4o',
    domain: 'medical',
    accuracy_mean: 0.0,
    accuracy_ci_lower: 0.0,
    accuracy_ci_upper: 0.0,
    hallucination_rate_mean: 0.0,
    grounding_score_mean: 0.0,
    avg_cost_usd: 0.0,
    p95_latency_ms: 0,
    ci_gate_passed: false,
  },
]

function computeParetoFrontier(points: ParetoPoint[]): ParetoPoint[] {
  // A point is Pareto-optimal if no other point has both higher quality AND lower cost
  return points.map(p => {
    const dominated = points.some(
      other =>
        other.model !== p.model &&
        other.quality >= p.quality &&
        other.cost <= p.cost &&
        (other.quality > p.quality || other.cost < p.cost)
    )
    return { ...p, isPareto: !dominated }
  })
}

export default function Home() {
  const [runs, setRuns] = useState<LeaderboardEntry[]>(SEED_RUNS.filter(r => r.accuracy_mean > 0))
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [domainFilter, setDomainFilter] = useState<string>('all')
  const [loading, setLoading] = useState(false)

  // Try to fetch live data from FastAPI — falls back to seed data gracefully
  useEffect(() => {
    const fetchLive = async () => {
      try {
        const res = await fetch('/api/backend/leaderboard', { signal: AbortSignal.timeout(3000) })
        if (res.ok) {
          const data = await res.json()
          if (data.length > 0) setRuns(data)
        }
      } catch {
        // API not running — seed data already shown
      }
    }
    fetchLive()
  }, [])

  const filtered = domainFilter === 'all' ? runs : runs.filter(r => r.domain === domainFilter)

  const paretoData: ParetoPoint[] = computeParetoFrontier(
    filtered.map(r => ({
      model: r.model_display_name,
      cost: r.avg_cost_usd * 1000, // per 1K calls for readability
      quality: r.accuracy_mean * 100,
      latency: r.p95_latency_ms,
      isPareto: false,
    }))
  )

  const bestRun = filtered.reduce<LeaderboardEntry | null>(
    (best, r) => (!best || r.accuracy_mean > best.accuracy_mean ? r : best),
    null
  )

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>

      {/* ── Header ── */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        padding: '0 2rem',
        height: '56px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        background: 'var(--bg-base)',
        zIndex: 100,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: 'var(--accent-green)',
            boxShadow: '0 0 6px var(--accent-green)',
            animation: 'pulse 2s infinite',
          }} />
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.08em' }}>
            GENAI RELIABILITY FRAMEWORK
          </span>
        </div>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
            OSFI E-23 ALIGNED
          </span>
          <span style={{ width: 1, height: 16, background: 'var(--border)' }} />
          <a
            href="https://github.com/CrillyPienaah/genai-reliability-framework"
            target="_blank"
            rel="noreferrer"
            style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}
          >
            GitHub ↗
          </a>
        </div>
      </header>

      <main style={{ padding: '2rem', maxWidth: 1200, margin: '0 auto' }}>

        {/* ── Title block ── */}
        <div style={{ marginBottom: '2rem', animationDelay: '0ms' }} className="animate-in">
          <h1 style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 28,
            fontWeight: 600,
            color: 'var(--text-primary)',
            letterSpacing: '-0.02em',
            marginBottom: 8,
          }}>
            Model Evaluation Leaderboard
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: 14, maxWidth: 680, lineHeight: 1.7 }}>
            Domain-grounded reliability benchmarks for regulated medical and financial workflows.
            All metrics include 95% bootstrapped confidence intervals (n=1,000 resamples).
            Grounding verified deterministically against source documents before LLM judge scoring.
          </p>
        </div>

        {/* ── Summary cards ── */}
        {bestRun && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 12,
            marginBottom: '2rem',
            animationDelay: '80ms',
          }} className="animate-in">
            <MetricCard
              label="Best accuracy"
              value={`${(bestRun.accuracy_mean * 100).toFixed(1)}%`}
              sub={`CI [${(bestRun.accuracy_ci_lower * 100).toFixed(1)}%, ${(bestRun.accuracy_ci_upper * 100).toFixed(1)}%]`}
              accent="green"
            />
            <MetricCard
              label="Best model"
              value={bestRun.model_display_name}
              sub={bestRun.domain}
              accent="blue"
            />
            <MetricCard
              label="Cost / call"
              value={`$${bestRun.avg_cost_usd.toFixed(4)}`}
              sub="avg generation only"
              accent="amber"
            />
            <MetricCard
              label="Grounding score"
              value={`${(bestRun.grounding_score_mean * 100).toFixed(0)}%`}
              sub="deterministic gate"
              accent="teal"
            />
            <MetricCard
              label="Cases evaluated"
              value={`${filtered.reduce((s, r) => s + 30, 0)}`}
              sub={`${filtered.length} model run${filtered.length !== 1 ? 's' : ''}`}
              accent="none"
            />
          </div>
        )}

        {/* ── Domain filter ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: '1.5rem', alignItems: 'center' }}>
          <span style={{ color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 11, letterSpacing: '0.08em' }}>
            DOMAIN
          </span>
          {['all', 'medical', 'finance'].map(d => (
            <button
              key={d}
              onClick={() => setDomainFilter(d)}
              style={{
                padding: '4px 14px',
                borderRadius: 4,
                border: '1px solid',
                borderColor: domainFilter === d ? 'var(--accent-blue)' : 'var(--border)',
                background: domainFilter === d ? 'var(--accent-blue-dim)' : 'transparent',
                color: domainFilter === d ? 'var(--accent-blue)' : 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                cursor: 'pointer',
                letterSpacing: '0.06em',
                transition: 'all 0.15s',
              }}
            >
              {d.toUpperCase()}
            </button>
          ))}
        </div>

        {/* ── Leaderboard table ── */}
        <section style={{ marginBottom: '2.5rem' }} className="animate-in">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <h2 style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.08em', fontWeight: 500 }}>
              RESULTS — {domainFilter.toUpperCase()} DOMAIN
            </h2>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
              {filtered.length} run{filtered.length !== 1 ? 's' : ''}
            </span>
          </div>
          <LeaderboardTable
            entries={filtered}
            selectedId={selectedRun}
            onSelect={setSelectedRun}
          />
        </section>

        {/* ── Two column: Pareto + detail ── */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: filtered.length > 1 ? '1fr 1fr' : '1fr',
          gap: '1.5rem',
          marginBottom: '2.5rem',
        }}>
          {filtered.length >= 1 && (
            <section className="animate-in">
              <h2 style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.08em', fontWeight: 500, marginBottom: 12 }}>
                COST–QUALITY PARETO FRONTIER
              </h2>
              <div style={{ background: 'var(--bg-surface)', border: '1px solid var(--border)', borderRadius: 8, padding: '1.5rem' }}>
                <ParetoChart data={paretoData} />
                <p style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 12, fontFamily: 'var(--font-mono)' }}>
                  X: cost per 1K calls (USD) · Y: accuracy (%) · Pareto-optimal models highlighted
                </p>
              </div>
            </section>
          )}

          {selectedRun && (
            <section className="animate-in">
              <h2 style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-secondary)', letterSpacing: '0.08em', fontWeight: 500, marginBottom: 12 }}>
                RUN DETAIL
              </h2>
              <RunDetails entry={filtered.find(r => r.run_id === selectedRun)!} />
            </section>
          )}
        </div>

        {/* ── OSFI E-23 note ── */}
        <div style={{
          border: '1px solid var(--border)',
          borderLeft: '3px solid var(--accent-amber)',
          borderRadius: 4,
          padding: '1rem 1.25rem',
          background: 'var(--bg-surface)',
          marginBottom: '2rem',
        }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent-amber)', letterSpacing: '0.08em', marginBottom: 6 }}>
            OSFI E-23 — MODEL RISK MANAGEMENT ALIGNMENT
          </div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.7 }}>
            Canadian federally regulated financial institutions must treat generative AI outputs as model outputs under
            OSFI Guideline E-23 (in force May 2027). This framework addresses the core validation challenge:
            non-deterministic systems cannot be validated with deterministic tests. Bootstrapped confidence intervals
            with statistical significance gates, combined with deterministic mechanistic grounding checks, provide
            the traceable, auditable validation evidence OSFI E-23 requires.
          </p>
        </div>

        {/* ── Footer ── */}
        <footer style={{ borderTop: '1px solid var(--border)', paddingTop: '1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
            GenAI Reliability Framework · Christopher Dankwa · Northeastern University MPS Analytics 2026
          </span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
            <a href="https://github.com/CrillyPienaah/genai-reliability-framework" target="_blank" rel="noreferrer">
              github.com/CrillyPienaah/genai-reliability-framework
            </a>
          </span>
        </footer>
      </main>
    </div>
  )
}


