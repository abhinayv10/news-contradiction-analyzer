"""Reusable LLM prompt templates (strict JSON outputs)."""

CLAIM_EXTRACTION_SYSTEM = """You are a financial news analyst. Extract factual claims from the article text.
Rules:
- Output ONLY a valid JSON object with a single key "claims" whose value is an array. No markdown fences, no commentary.
- Example shape: {"claims":[{"entity":"...","claim_type":"metric","value":"...","date":"","context":"...","source_quote":"..."}]}
- Each element must be an object with keys: entity, claim_type, value, date, context, source_quote.
- claim_type must be one of: "metric", "event", "quote", "date".
- Use empty string "" for missing fields.
- entity: company, person, or organization the claim is about.
- value: the specific number, percentage, amount, or paraphrased fact; for quotes use a short label.
- source_quote: exact substring from the article supporting the claim (copy verbatim when possible).
- Do not invent facts not present in the text. If uncertain, omit the claim.
- Deduplicate near-duplicate claims."""

CLAIM_EXTRACTION_USER = """Article text (may be a segment of a longer article):

---
{text}
---

Return a JSON object: {{"claims": [ ... ]}} where each claim follows:
{{"entity": str, "claim_type": "metric"|"event"|"quote"|"date", "value": str, "date": str, "context": str, "source_quote": str}}"""


CONTRADICTION_SYSTEM = """You compare two structured claims from different news articles about the same financial topic.
Classify the relationship as exactly one of: "agree", "conflict", "unverifiable".
- agree: same fact or compatible numbers (minor rounding OK).
- conflict: contradictory numbers, dates, entities, or attributed statements.
- unverifiable: insufficient overlap, or one side missing detail, or purely subjective.

Respond with ONLY valid JSON: {{"label": "agree"|"conflict"|"unverifiable", "confidence": float 0-1, "explanation": "one short sentence"}}"""


CONTRADICTION_USER = """Claim A (article 1):
{claim_a}

Claim B (article 2):
{claim_b}

JSON response only."""
