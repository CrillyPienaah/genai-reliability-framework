export interface BootstrappedMetric {
  mean: number
  ci_lower: number
  ci_upper: number
  n_samples: number
}

export interface ModelConfig {
  provider: string
  model_id: string
  display_name: string
  temperature: number
  max_tokens: number
}

export interface RunSummary {
  run_id: string
  model_cfg: ModelConfig
  domain: string
  n_cases: number
  accuracy: BootstrappedMetric
  hallucination_rate: BootstrappedMetric
  grounding_score: BootstrappedMetric
  avg_cost_usd: BootstrappedMetric
  p50_latency_ms: number
  p95_latency_ms: number
  ci_gate_passed: boolean
  baseline_run_id: string | null
  created_at: string
}

export interface LeaderboardEntry {
  run_id: string
  model_display_name: string
  model_id: string
  domain: string
  accuracy_mean: number
  accuracy_ci_lower: number
  accuracy_ci_upper: number
  hallucination_rate_mean: number
  grounding_score_mean: number
  avg_cost_usd: number
  p95_latency_ms: number
  ci_gate_passed: boolean
}

// Pareto chart data point
export interface ParetoPoint {
  model: string
  cost: number
  quality: number
  latency: number
  isPareto: boolean
}
