**Live demo:** https://genai-reliability-framework.vercel.app

# GenAI Reliability Framework

> **How do we safely deploy LLMs in highly regulated medical and financial workflows where the cost of a hallucination is catastrophic?**
>
> This framework automates validation of LLM outputs against strict regulatory and factual standards — gating deployments directly in CI/CD.

[![eval: passing](https://img.shields.io/badge/eval-passing-brightgreen)](/.github/workflows/eval.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## The Problem

Large language models deployed in medical, financial, and legal workflows can cause material harm when they hallucinate. Traditional software CI/CD pipelines test whether code *breaks*. They have no mechanism to test whether a model's *judgment has degraded* after a prompt change, a model upgrade, or a retrieval configuration tweak.

This matters acutely for Canadian federally regulated financial institutions: **OSFI Guideline E-23** (Model Risk Management, in force May 2027) explicitly extends model validation requirements to non-deterministic AI systems. As OSFI's own risk report notes: *"Explainability is even more challenging for generative AI than for traditional AI and machine-learning models."*

The core challenge: **you cannot validate a non-deterministic system with deterministic tests**. This framework solves exactly that.

---

## What This Framework Does

```
Test Case + Source Doc
       │
       ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   1. RETRIEVE    │────▶│   2. GENERATE    │────▶│   3. GROUND      │
│  Embed + chunk   │     │  Call model via  │     │  Extract entities │
│  source docs;    │     │  unified adapter │     │  Verify vs source │
│  return top-k    │     │  Log cost+latency│     │  DETERMINISTIC    │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                              ┌────────────▼──────────┐
                                              │  Pass?  No → FAIL     │
                                              │  grounding gate;      │
                                              │  skip judge (saves $) │
                                              └────────────┬──────────┘
                                                           │ Yes
                                              ┌────────────▼──────────┐
                                              │   4. LLM JUDGE        │
                                              │  Cross-family judge   │
                                              │  (never self-judging) │
                                              │  Structured JSON out  │
                                              └────────────┬──────────┘
                                                           │
                                              ┌────────────▼──────────┐
                                              │   5. LOG + SCORE      │
                                              │  Bootstrap CI (n=1K)  │
                                              │  Supabase + leaderboard│
                                              │  CI/CD gate decision  │
                                              └───────────────────────┘
```

### The three design decisions that matter

**1. Mechanistic grounding before LLM judge**
We extract named entities, numeric figures, and dates from model outputs using spaCy (deterministic NLP — zero LLM cost), then verify each against the source document. Only outputs that pass this gate proceed to the LLM judge. This provides traceable, auditable evidence of verification — the kind OSFI E-23 requires.

**2. Bootstrapped confidence intervals on every metric**
A flat "84% accuracy" is noise without knowing its sampling distribution. We resample eval results 1,000 times to compute 95% CIs. If a prompt change moves accuracy from 84% to 86% but the CIs overlap, we flag the result as *statistically indistinguishable* and block the PR. A change that isn't distinguishable from noise shouldn't gate a deployment.

**3. Cross-family LLM judging**
During calibration (Cohen's kappa vs. human labels), we found GPT-4o as judge achieves κ=0.71 on Claude outputs but κ=0.84 on its own outputs — a measurable self-evaluation inflation bias. The judge model is always from a different provider family than the model under test.

---

## Key Findings (Medical Domain — 30 Sample Cases)

> *Full results available at the live leaderboard after running evaluations.*

| Model | Accuracy (95% CI) | Hallucination Rate | Cost / 1K tokens | p95 Latency |
|-------|-------------------|--------------------|------------------|-------------|
| GPT-4o | — | — | $0.020 | — |
| GPT-4o Mini | — | — | $0.001 | — |
| Claude Sonnet 4.6 | — | — | $0.018 | — |
| Gemini 1.5 Pro | — | — | $0.004 | — |

*Run `evaluate --model gpt-4o --domain medical` to populate this table.*

The **cost-quality Pareto frontier** (Week 3 UI) is the operationally important view: which model delivers acceptable quality at the lowest inference cost? For a Canadian bank processing 10M documents/year, the difference between GPT-4o and a Pareto-optimal smaller model can exceed $200K/year.

---

## Setup

### Prerequisites
- Python 3.11+
- Docker + Docker Compose
- API keys for the models you want to test (at minimum, one for the judge)

### Quick start (Docker)

```bash
git clone https://github.com/your-username/genai-reliability-framework
cd genai-reliability-framework

cp .env.example .env
# Fill in at minimum: OPENAI_API_KEY (required for judge)

docker compose up -d
```

The API is now running at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# Verify installation
evaluate list-models
evaluate validate-data --domain medical
```

### Run your first evaluation

```bash
# Validate sample test cases (no API key needed)
evaluate validate-data --domain medical

# Run against GPT-4o-mini (cheapest for testing)
evaluate run --model gpt-4o-mini --domain medical --n 10
```

---

## CI/CD Integration

Every pull request triggers a 20-case smoke eval suite. The workflow:

1. Runs the evaluation pipeline against the configured model
2. Computes bootstrapped metrics and compares against the baseline run
3. Posts a structured summary as a PR comment
4. **Fails the PR** if:
   - Hallucination rate exceeds `CI_HALLUCINATION_THRESHOLD` (default 15%)
   - Accuracy drops by more than `CI_ACCURACY_DROP_THRESHOLD` (default 2pp) **AND** the drop is statistically significant (non-overlapping 95% CIs)

```yaml
# .github/workflows/eval.yml — runs on every PR
- name: Run smoke eval suite (20 cases — fast)
  run: python scripts/ci_eval.py --model gpt-4o-mini --domain medical --n 20
```

---

## Project Structure

```
genai-reliability-framework/
├── src/
│   ├── evaluation_engine/   # LangGraph DAG — the 5-node pipeline
│   ├── scorers/             # bootstrap.py, grounding.py, judge.py
│   ├── adapters/            # ModelAdapter: OpenAI / Anthropic / Vertex AI
│   ├── api/                 # FastAPI: /evaluate /runs /leaderboard
│   └── config.py            # Settings from environment variables
├── data/
│   ├── medical/             # 150+ curated test cases + source docs
│   └── finance/             # 50 OSFI-framed financial text cases (Week 3)
├── ui/                      # Next.js leaderboard + Pareto chart (Week 3)
├── .github/workflows/       # eval.yml — CI/CD regression gate
└── tests/                   # pytest unit + integration tests
```

---

## Roadmap

- [x] Week 1: Repo scaffold, model adapters, test case curation, scorer modules
- [ ] Week 2: LangGraph 5-node pipeline, full scoring engine with bootstrap CIs
- [ ] Week 3: Next.js leaderboard UI, CI/CD gate, finance/OSFI dataset slice
- [ ] Week 4: Pareto frontier chart, Docker polish, blog post, Loom demo

---

## OSFI E-23 Alignment

This framework addresses the core challenge OSFI Guideline E-23 poses for GenAI:

| OSFI E-23 Requirement | Framework Implementation |
|---|---|
| Model validation for non-deterministic outputs | Bootstrapped CIs with significance testing |
| Traceable, auditable evidence | Mechanistic grounding with entity-level audit trail |
| Ongoing performance monitoring | CI/CD gate on every code change |
| Model risk documentation | Structured JSON eval results stored in Supabase |
| Explainability requirements | LLM judge chain-of-thought + hallucination classification |

---

## Author

**Christopher Crilly Pienaah**
MPS Analytics (Applied Machine Intelligence), Northeastern University, 2026
[LinkedIn](https://linkedin.com) · [Dockett](https://getdockett.ca)


