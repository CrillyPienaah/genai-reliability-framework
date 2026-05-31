"""
src/scorers/grounding.py
─────────────────────────
Deterministic mechanistic grounding checks.

WHY THIS EXISTS (interview talking point):
  In regulated industries, "LLM-as-judge" alone is treated with deep scepticism
  because it introduces a second layer of non-determinism. Before any LLM judge
  sees the output, we run deterministic checks:

    1. Extract named entities, numeric figures, and dates from the model output
       using pure Python regex — zero compiled dependencies, works on any Python version.
    2. Verify each extracted entity is present in the source document
       via exact string match OR token-overlap fuzzy match.
    3. Emit a grounding_score = verified / extracted.

  Only outputs that PASS the deterministic gate proceed to the LLM judge.
  This aligns with OSFI E-23's requirement for traceable, auditable
  model validation evidence.

OSFI E-23 alignment:
  "Model outputs must be traceable to documented inputs" — our grounding
  check provides an audit trail of which claims were verified against source.
"""

from __future__ import annotations

import re

import structlog

from src.models import GroundingResult, HallucinationType

logger = structlog.get_logger(__name__)

GROUNDING_THRESHOLD = 0.70  # grounding_score must exceed this to pass


# ── Entity extraction (pure Python — no compiled deps) ────────────────────────


def extract_entities(text: str) -> list[str]:
    """
    Pull numeric figures, dosages, dates, codes, and key terms from text.
    Pure regex — no spaCy or torch required. Works on Python 3.11+.

    Returns a deduplicated list of entity strings.
    """
    if not text:
        return []

    entities: set[str] = set()

    patterns = [
        # Medical dosages: 500 mg, 4000 mg, 0.3 mcg, etc.
        r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|L|mmol|mmHg|bpm|mEq|ng|IU)\b",
        # Large numbers with commas: 4,000 / 10,000
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        # Plain integers 2+ digits
        r"\b\d{2,}\b",
        # Medical / regulatory codes
        r"\b(?:ICD|CPT|NDC|DSM)-?\d+\b",
        r"\bOSFI\s+E-\d+\b",
        # Date patterns
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
        r"Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        # Percentages
        r"\b\d+(?:\.\d+)?\s*%",
        # Clinical score names
        r"\b(?:INR|eGFR|HbA1c|TSH|SOFA|qSOFA|MMSE|MAP)\b",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            val = match.group().strip()
            if len(val) >= 2:
                entities.add(val)

    return sorted(entities)


# ── Verification ──────────────────────────────────────────────────────────────


def _exact_match(entity: str, source: str) -> bool:
    return entity.lower() in source.lower()


def _fuzzy_match(entity: str, source_sentences: list[str]) -> bool:
    """
    Token-overlap fuzzy match — all tokens in entity must appear
    in at least one source sentence. No ML required.
    """
    tokens = entity.lower().split()
    for sentence in source_sentences:
        sent_lower = sentence.lower()
        if all(t in sent_lower for t in tokens):
            return True
    return False


def verify_entities(extracted: list[str], source_text: str) -> list[str]:
    """
    Returns the subset of extracted entities verified in source_text.
    Tries exact match first (cheap), then fuzzy match.
    """
    source_sentences = [s.strip() for s in source_text.split(".") if s.strip()]
    verified: list[str] = []

    for entity in extracted:
        if _exact_match(entity, source_text):
            verified.append(entity)
        elif _fuzzy_match(entity, source_sentences):
            verified.append(entity)

    return verified


def infer_hallucination_type(
    extracted: list[str],
    verified: list[str],
    model_output: str,
    expected_answer: str,
) -> HallucinationType:
    """Classify the type of hallucination when grounding fails."""
    if not extracted:
        return HallucinationType.NONE

    unverified = set(extracted) - set(verified)
    if not unverified:
        return HallucinationType.NONE

    # Missing expected facts → omission
    expected_entities = extract_entities(expected_answer)
    output_lower = model_output.lower()
    missing_expected = [e for e in expected_entities if e.lower() not in output_lower]
    if len(missing_expected) > len(expected_entities) * 0.3:
        return HallucinationType.OMISSION

    # Negation near entity → contradiction
    negation_pattern = re.compile(r"\b(?:not|never|no|isn't|aren't|wasn't)\b", re.I)
    if negation_pattern.search(model_output):
        return HallucinationType.CONTRADICTION

    return HallucinationType.FABRICATION


# ── Main entry point ──────────────────────────────────────────────────────────


def run_grounding_check(
    model_output: str,
    source_text: str,
    expected_answer: str,
    threshold: float = GROUNDING_THRESHOLD,
) -> GroundingResult:
    """
    Full mechanistic grounding check — called as a LangGraph node.
    """
    if not model_output.strip():
        logger.warning("grounding_check_empty_output")
        return GroundingResult(
            extracted_entities=[],
            verified_entities=[],
            grounding_score=0.0,
            hallucination_type=HallucinationType.FABRICATION,
            deterministic_pass=False,
        )

    extracted = extract_entities(model_output)

    if not extracted:
        # No extractable entities — pass to LLM judge for semantic evaluation
        logger.debug("grounding_no_entities_extracted", output_len=len(model_output))
        return GroundingResult(
            extracted_entities=[],
            verified_entities=[],
            grounding_score=1.0,
            hallucination_type=HallucinationType.NONE,
            deterministic_pass=True,
        )

    verified = verify_entities(extracted, source_text)
    score = len(verified) / len(extracted)

    hallucination_type = infer_hallucination_type(
        extracted, verified, model_output, expected_answer
    )

    result = GroundingResult(
        extracted_entities=extracted,
        verified_entities=verified,
        grounding_score=round(score, 4),
        hallucination_type=hallucination_type,
        deterministic_pass=score >= threshold,
    )

    logger.info(
        "grounding_check_complete",
        n_extracted=len(extracted),
        n_verified=len(verified),
        score=result.grounding_score,
        passed=result.deterministic_pass,
    )

    return result
