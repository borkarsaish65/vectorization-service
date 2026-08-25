# Release 2.1.0 — Acronym-Aware Search

**Service:** vectorization-service

> ⚠️ **Run the migration before starting the service on a fresh deploy.** The acronym cache warms at startup by querying `acronym_mapping` — if that table doesn't exist yet, warm-up fails (non-fatal, logged, service still boots) and acronym detection silently returns nothing until the migration runs and the cache is repopulated.

---

## Table of Contents

- [What's New](#whats-new)
- [Acronym Search Architecture](#acronym-search-architecture)
- [Dependencies](#dependencies)
- [Deployment](#deployment)
- [Rollback](#rollback)

---

## What's New

- **Acronym detection in search queries** — a query containing a known acronym (e.g. `"SSC"`, `"PTM"`) is detected and expanded before retrieval, so a search for the acronym also matches documents that only spell out its full meaning, and vice versa. Gated by `ACRONYM_SEARCH_ENABLED` (defaults to `true` this release — validated on real traffic).
- **Acronym dictionary** — new `acronym_mapping` Postgres table (multi-expansion aware: one acronym can map to more than one meaning, e.g. `SSC` → "Staff Selection Commission" / "Sainik School Society"), backed by a Redis cache-aside layer (`get_expansions_batch`) so repeat lookups don't hit Postgres.
- **Bulk upload endpoint** — `POST /api/acronyms/bulk`, internal-only (`X-Internal-Token` header), CSV upload to create/update/**enable or disable** acronyms in batch. See [CSV format](#bulk-upload-csv-format) below.
- **Stopword filtering on the detection path** — a multi-word query's all-caps filler tokens (`THE`, `AND`, `ON`, …) are dropped before the Redis/DB lookup, cutting wasted round-trips. Reuses spaCy's built-in stopword list — no new dependency, no extra model download.
- **`acronym_info` in the search response** — when an acronym is detected, the response now includes `{"detected": true, "mapping": {"SSC": ["Staff Selection Commission", "Sainik School Society"]}}`; `null` when nothing was detected.

---

## Acronym Search Architecture

<details>
<summary>Detect → expand → retrieve flow</summary>

1. **Detect** (`acronym_query_service.detect_acronyms`) — tokenizes the query; a fully-uppercase token (or the whole query, if it's a single word) is checked against the dictionary. Lowercase/mixed-case words inside a longer query are skipped (precision guard against accidental collisions, e.g. `"diet"` in `"the diet chart"`).
2. **Look up** (`acronym_service.get_expansions_batch`) — one batched Redis round-trip for every candidate token in the query, falling back to Postgres (and writing through to Redis) for whatever's still missing. Never one round-trip per token.
3. **Expand and retrieve** (inline in `PrioritizedSearchService.search()`) — a detected acronym produces up to 2 dense embeddings (original query + one substituted with the acronym's primary expansion, never concatenated into one string) and 1 combined sparse/BM25 query (original text plus **every** expansion's words appended). `_parallel_batch_search`/`_hybrid_batch_search` accept a list of embeddings and merge per-field with `max()`.

</details>

<details>
<summary>Acronym-aware ranking</summary>

For an acronym-detected query, a document with a genuine sparse/keyword hit is scored both with the normal dense/sparse blend and with the blend flipped to trust sparse more — whichever is higher wins. A real keyword match can only raise a document's score, never lower it. This does not change ranking for non-acronym queries.

</details>

<details>
<summary>Cache-aside layer</summary>

- `get_expansions_batch` — cache-aside read: Redis hit returns immediately; miss falls back to Postgres and writes through. Negative-caches "not an acronym" (`REDIS_NEGATIVE_CACHE_TTL`) and DB-outage misses (`REDIS_DB_ERROR_CACHE_TTL`, shorter — a DB outage is usually transient).
- `refresh_cache` — re-caches exactly the acronyms touched by a bulk upload; a row set inactive gets its cache entry **deleted**, not refreshed, so disabling takes effect immediately rather than waiting out the TTL.
- `invalidate_cache` — batched delete (single Redis `DEL` for N keys), used as the fallback if `refresh_cache` itself fails.
- `load_acronym_cache` — warms the whole dictionary at startup only; per-row error handling so one bad Redis write doesn't lose the rest of the warm-up.

</details>

<details>
<summary>Bulk upload CSV format</summary>

| Column | Required | Notes |
|---|---|---|
| `acronym` | Yes | Case-insensitive, stored uppercase. Max 32 chars. |
| `expansions` | Yes | Pipe-separated if more than one meaning, e.g. `Staff Selection Commission\|Sainik School Society`. |
| `description` | No | Free text. |
| `is_active` | No | `true`/`false`, case-insensitive. Missing/empty defaults to active (backward-compatible with CSVs from before this column existed). Any other value is rejected as a per-row error. |

One invalid or duplicate-in-batch row doesn't fail the whole upload — it's reported in the response's `errors` list while the rest of the batch still commits.

</details>

---

## Dependencies

- **No new packages.** `alembic`, `sqlalchemy`, `psycopg2-binary` were already in `requirements.txt` from the earlier PRs in this stack (schema/cache work); `spacy>=3.7.0` was already required for existing query preprocessing.
- **No new model download.** The stopword filter uses `spacy.lang.en.stop_words.STOP_WORDS` — a static list bundled with the `spacy` package itself, distinct from the `en_core_web_sm` pipeline model already required elsewhere (that one *is* a separate download, unrelated to this release).
- **`data/acronyms.csv`** (553 seed acronyms) is committed in the repo — the migration reads it directly, no separate upload step needed for the initial seed.

---

## Deployment

### Pre-deploy checklist

1. **Verify Postgres is reachable** — `acronym_mapping` is a new table; the existing `translations` table is untouched.
2. **Set `INTERNAL_API_TOKEN`** in `.env` to a real secret — gates `POST /api/acronyms/bulk`. No default; an unset token rejects every request to that endpoint (not a soft-fail).

### Deploy steps

3. Add/confirm the new env keys in `.env` (see `.env.sample`):
   ```dotenv
   ACRONYM_SEARCH_ENABLED=true
   ACRONYM_BULK_UPLOAD_MAX_SIZE_MB=5
   INTERNAL_API_TOKEN=<generate-a-real-secret>
   ```
   Optional, sensible defaults if omitted: `REDIS_CACHE_TTL=86400`, `REDIS_NEGATIVE_CACHE_TTL=3600`, `REDIS_DB_ERROR_CACHE_TTL=30`, `REDIS_SOCKET_CONNECT_TIMEOUT=1`, `REDIS_SOCKET_TIMEOUT=1`.

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the migration** — creates `acronym_mapping` and seeds it from `data/acronyms.csv` in one step. Idempotent (`ON CONFLICT DO UPDATE`) — safe to re-run:
   ```bash
   alembic upgrade head
   ```

6. **Start the service.** The acronym cache warms automatically at startup — check the log line:
   ```
   Acronym cache warmed: 553 active acronym(s)
   ```
   If this logs a warning instead ("Acronym cache warm-up failed, continuing without it") the service still boots — acronym detection just falls back to per-lookup Postgres queries until the cache naturally repopulates.

7. **(Optional) Upload or update additional acronyms** later via the bulk endpoint:
   ```bash
   curl -X POST "http://<HOST>:<PORT>/api/acronyms/bulk" \
     -H "X-Internal-Token: <INTERNAL_API_TOKEN>" \
     -F "file=@acronyms.csv;type=text/csv"
   ```

---

## Rollback

**Fastest option — flag off, no code/schema revert:** set `ACRONYM_SEARCH_ENABLED=false` and restart. Detection never runs; `search()` falls through to exactly its pre-acronym-feature behavior (dense_query_texts/sparse_query_text stay single-string, no reweighting, `acronym_info` is always `null`).

**Full schema rollback** (only if the table itself needs to go): `alembic downgrade -1` drops `acronym_mapping` entirely. ⚠️ **Destructive** — any acronym added or edited via the bulk endpoint after the initial seed is lost, not just reverted to the seed state. Take a Postgres backup first if that data matters.

A code revert of this branch is not expected to be necessary on its own — the retrieval-layer changes (merging the acronym multi-embedding fan-out into the existing search methods) were verified byte-identical to pre-existing behavior for the non-acronym path, independent of the feature flag.
