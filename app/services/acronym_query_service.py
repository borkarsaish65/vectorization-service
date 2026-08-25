import re
from typing import Dict, List

from spacy.lang.en.stop_words import STOP_WORDS

from app.services.acronym_service import get_expansions_batch

# Letters only — strips internal dots ("D.I.E.T." -> "DIET"), digits, and any
# surrounding punctuation in one pass. Known simplification: a token like
# "PTM2024" also normalizes to "PTM", which isn't addressed by the spec.
_NON_LETTER_RE = re.compile(r"[^A-Za-z]")


def _normalize_token(raw_token: str) -> str:
    return _NON_LETTER_RE.sub("", raw_token)


def detect_acronyms(query: str) -> Dict[str, List[str]]:
    """Detect acronyms in `query` via one batched Redis/Postgres lookup covering
    every candidate token. Must be called with the case-preserving query, not
    the lowercased/preprocessed one — these rules depend on case.

    Case rules:
      - A fully-uppercase token (length > 1) is always checked.
      - A single-word query is checked regardless of case.
      - A lowercase/mixed-case word inside a multi-word query is skipped (e.g.
        "diet" in "the diet chart" must not match the DIET acronym).
      - An uppercase stopword (THE, AND, ON) in a multi-word query is dropped
        before lookup; single-word queries are exempt so "BE" stays searchable.
    """
    if not query or not query.strip():
        return {}

    raw_tokens = query.strip().split()
    is_single_word_query = len(raw_tokens) == 1

    # Collect every case/length-filtered candidate first, deduped, before any
    # lookup — as opposed to deduping only successfully-resolved acronyms
    # (which would mean two occurrences of the same non-acronym uppercase
    # word each triggered their own lookup).
    candidates: List[str] = []
    seen = set()
    for raw_token in raw_tokens:
        normalized = _normalize_token(raw_token)
        if len(normalized) < 2:
            continue

        if not (normalized.isupper() or is_single_word_query):
            continue

        if not is_single_word_query and normalized.lower() in STOP_WORDS:
            continue

        candidate = normalized.upper()
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)

    if not candidates:
        return {}

    expansions_by_acronym = get_expansions_batch(candidates)
    # `if expansions` (falsy, not just "in the dict") rejects an empty-list
    # expansions value — the expansions column defaults to '[]'::jsonb and
    # only bulk_upsert() validates non-empty before insert, so a row written
    # via any other path (raw SQL, a future writer) could still have
    # expansions=[]. Guarding here closes both downstream call sites in
    # search() (the acronym-substitution regex, the sparse OR-join) at once —
    # neither ever sees an empty-list acronym in its mapping.
    return {
        acronym: expansions
        for acronym, expansions in expansions_by_acronym.items()
        if expansions
    }
