-- scripts/init_db.sql
-- Supabase-compatible schema for the GenAI Reliability Framework.
-- Run automatically by docker-compose on first start.
-- Also apply via: supabase db push (for cloud Supabase).

-- ── Extensions ──────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- for fast text search on outputs

-- ── Enums ───────────────────────────────────────────────────────────────────
CREATE TYPE domain_enum AS ENUM ('medical', 'finance', 'legal');
CREATE TYPE severity_enum AS ENUM ('critical', 'high', 'medium', 'low');
CREATE TYPE scenario_type_enum AS ENUM ('happy_path', 'adversarial', 'recoverable_error', 'out_of_scope');
CREATE TYPE hallucination_type_enum AS ENUM ('fabrication', 'contradiction', 'omission', 'none');
CREATE TYPE model_provider_enum AS ENUM ('openai', 'anthropic', 'google');

-- ── Test cases ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS test_cases (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    domain          domain_enum NOT NULL,
    severity        severity_enum NOT NULL,
    scenario_type   scenario_type_enum NOT NULL,
    prompt          TEXT NOT NULL,
    expected_answer TEXT NOT NULL,
    source_doc_id   TEXT NOT NULL,
    tags            TEXT[] DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_test_cases_domain ON test_cases(domain);
CREATE INDEX idx_test_cases_severity ON test_cases(severity);

-- ── Eval runs ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_runs (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_provider      model_provider_enum NOT NULL,
    model_id            TEXT NOT NULL,
    model_display_name  TEXT NOT NULL,
    domain              domain_enum NOT NULL,
    n_cases             INTEGER NOT NULL DEFAULT 0,
    -- Bootstrapped accuracy
    accuracy_mean       FLOAT,
    accuracy_ci_lower   FLOAT,
    accuracy_ci_upper   FLOAT,
    -- Bootstrapped hallucination rate
    hallucination_mean  FLOAT,
    hallucination_ci_lower FLOAT,
    hallucination_ci_upper FLOAT,
    -- Bootstrapped grounding score
    grounding_mean      FLOAT,
    grounding_ci_lower  FLOAT,
    grounding_ci_upper  FLOAT,
    -- Cost / latency
    avg_cost_usd        FLOAT,
    p50_latency_ms      FLOAT,
    p95_latency_ms      FLOAT,
    -- CI gate
    ci_gate_passed      BOOLEAN DEFAULT TRUE,
    baseline_run_id     UUID REFERENCES eval_runs(id),
    pipeline_version    TEXT DEFAULT '0.1.0',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_eval_runs_domain ON eval_runs(domain);
CREATE INDEX idx_eval_runs_model ON eval_runs(model_id);
CREATE INDEX idx_eval_runs_created ON eval_runs(created_at DESC);

-- ── Per-response eval results ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS eval_results (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id                  UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    test_case_id            UUID NOT NULL REFERENCES test_cases(id),
    model_output            TEXT NOT NULL,
    -- Grounding
    extracted_entities      TEXT[] DEFAULT '{}',
    verified_entities       TEXT[] DEFAULT '{}',
    grounding_score         FLOAT NOT NULL,
    hallucination_type      hallucination_type_enum NOT NULL DEFAULT 'none',
    deterministic_pass      BOOLEAN NOT NULL,
    -- Judge
    accuracy_score          FLOAT NOT NULL,
    hallucination_detected  BOOLEAN NOT NULL,
    hallucination_explanation TEXT DEFAULT '',
    coherence_score         FLOAT NOT NULL,
    judge_confidence        FLOAT NOT NULL,
    judge_reasoning         TEXT,
    -- Metrics
    prompt_tokens           INTEGER NOT NULL,
    completion_tokens       INTEGER NOT NULL,
    total_tokens            INTEGER NOT NULL,
    latency_ms              FLOAT NOT NULL,
    cost_usd                FLOAT NOT NULL,
    evaluated_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_eval_results_run ON eval_results(run_id);
CREATE INDEX idx_eval_results_test_case ON eval_results(test_case_id);

-- ── Calibration samples ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS calibration_samples (
    id                          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_case_id                UUID NOT NULL REFERENCES test_cases(id),
    model_output                TEXT NOT NULL,
    human_accuracy_score        FLOAT NOT NULL CHECK (human_accuracy_score BETWEEN 0 AND 1),
    human_hallucination_label   BOOLEAN NOT NULL,
    human_grounding_score       FLOAT NOT NULL CHECK (human_grounding_score BETWEEN 0 AND 1),
    annotator_id                TEXT NOT NULL DEFAULT 'human_1',
    notes                       TEXT DEFAULT '',
    created_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- ── RLS policies (Supabase Row Level Security) ───────────────────────────────
-- Enable RLS — anon role can read, only service role can write.
ALTER TABLE test_cases        ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_runs         ENABLE ROW LEVEL SECURITY;
ALTER TABLE eval_results      ENABLE ROW LEVEL SECURITY;
ALTER TABLE calibration_samples ENABLE ROW LEVEL SECURITY;

-- Read-only for anon (powers the public leaderboard UI)
CREATE POLICY "anon_read_test_cases"    ON test_cases        FOR SELECT USING (true);
CREATE POLICY "anon_read_eval_runs"     ON eval_runs         FOR SELECT USING (true);
CREATE POLICY "anon_read_eval_results"  ON eval_results      FOR SELECT USING (true);

-- Write only via service role (API server)
CREATE POLICY "service_write_runs"    ON eval_runs    FOR INSERT WITH CHECK (true);
CREATE POLICY "service_write_results" ON eval_results FOR INSERT WITH CHECK (true);
CREATE POLICY "service_write_calibration" ON calibration_samples FOR INSERT WITH CHECK (true);

-- ── Leaderboard view (powers GET /leaderboard) ───────────────────────────────
CREATE OR REPLACE VIEW leaderboard AS
SELECT
    r.id                    AS run_id,
    r.model_display_name,
    r.model_id,
    r.domain,
    r.accuracy_mean,
    r.accuracy_ci_lower,
    r.accuracy_ci_upper,
    r.hallucination_mean    AS hallucination_rate_mean,
    r.grounding_mean        AS grounding_score_mean,
    r.avg_cost_usd,
    r.p95_latency_ms,
    r.ci_gate_passed,
    r.created_at
FROM eval_runs r
ORDER BY r.accuracy_mean DESC NULLS LAST;
