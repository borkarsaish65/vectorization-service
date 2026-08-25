import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.constants import (
    ACRONYM_CACHE_KEY_PREFIX,
    ACRONYM_CSV_COLUMN_ACRONYM,
    ACRONYM_CSV_COLUMN_DESCRIPTION,
    ACRONYM_CSV_COLUMN_EXPANSIONS,
    ACRONYM_CSV_COLUMN_IS_ACTIVE,
)
from app.core.clients import cache_client
from app.core.database import SessionLocal
from app.models.db_models import AcronymMapping

logger = logging.getLogger(__name__)


def _cache_key(acronym: str) -> str:
    return f"{ACRONYM_CACHE_KEY_PREFIX}:{acronym}"


def _valid_expansions(expansions) -> bool:
    """A DB/cache expansions value must be a non-empty list of non-empty
    strings. Guards against a malformed row (null, wrong type, empty
    strings) written outside bulk_upsert's validation — e.g. raw SQL —
    from being cached and crashing the acronym query-building code in
    prioritized_search_service.search() downstream."""
    return isinstance(expansions, list) and bool(expansions) and all(
        isinstance(e, str) and e.strip() for e in expansions
    )


def get_expansions_batch(acronyms: List[str]) -> Dict[str, List[str]]:
    """Cache-aside lookup for a list of acronyms in one call — Redis hits
    return immediately, misses fall back to Postgres and write through to
    the cache.

    Batched, not one round-trip per acronym: detect_acronyms() calls this
    once per query with every candidate token, which also collapses N
    independent connection-timeout exposures into one on a Redis outage.

    Input is deduped internally; acronyms not found (or empty) are simply
    absent from the returned dict."""
    normalized = list(dict.fromkeys(a.strip().upper() for a in acronyms if a and a.strip()))
    if not normalized:
        return {}

    keys = [_cache_key(a) for a in normalized]
    try:
        cached_values = cache_client.get_many(keys)
    except Exception as e:
        logger.warning(
            f"Acronym batch cache read failed for {len(normalized)} acronym(s), "
            f"falling back to Postgres: {e}"
        )
        cached_values = [None] * len(normalized)

    result: Dict[str, List[str]] = {}
    missing: List[str] = []
    for acronym, cached in zip(normalized, cached_values):
        if cached is None:
            missing.append(acronym)
            continue
        decoded = json.loads(cached)
        if decoded is not None:
            result[acronym] = decoded

    if not missing:
        return result

    db = SessionLocal()
    try:
        mappings = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym.in_(missing), AcronymMapping.is_active.is_(True))
            .all()
        )
    except Exception as e:
        # A Postgres outage (or the acronym table not existing yet) must
        # degrade to "not found" rather than propagating — search never
        # depended on Postgres before this feature. Negative-cache with the
        # short DB-error TTL so a Postgres outage doesn't get re-attempted
        # on every request for its full duration.
        logger.warning(
            f"Acronym batch DB lookup failed for {len(missing)} acronym(s), "
            f"treating as not found: {e}"
        )
        try:
            cache_client.set_many({
                _cache_key(a): (json.dumps(None), settings.REDIS_DB_ERROR_CACHE_TTL)
                for a in missing
            })
        except Exception as cache_e:
            logger.warning(f"Acronym batch DB-error negative-cache write failed: {cache_e}")
        return result
    finally:
        db.close()

    found_by_acronym = {}
    for mapping in mappings:
        if _valid_expansions(mapping.expansions):
            found_by_acronym[mapping.acronym] = mapping.expansions
        else:
            logger.warning(
                f"Acronym {mapping.acronym!r} has malformed expansions in DB "
                f"({mapping.expansions!r}), treating as not found"
            )
    cache_writes: Dict[str, Tuple[str, int]] = {}
    for acronym in missing:
        if acronym in found_by_acronym:
            result[acronym] = found_by_acronym[acronym]
            cache_writes[_cache_key(acronym)] = (
                json.dumps(found_by_acronym[acronym]), settings.REDIS_CACHE_TTL
            )
        else:
            cache_writes[_cache_key(acronym)] = (
                json.dumps(None), settings.REDIS_NEGATIVE_CACHE_TTL
            )

    try:
        cache_client.set_many(cache_writes)
    except Exception as e:
        logger.warning(
            f"Acronym batch cache write-through failed for {len(cache_writes)} acronym(s): {e}"
        )

    return result


def invalidate_cache(acronyms: List[str]) -> None:
    """Drop cached entries for the given acronyms in one batched round-trip.
    Call after any write to those rows so stale expansions aren't served
    until TTL expiry.

    Swallows Redis errors (logged) rather than raising — this is already the
    degraded-mode fallback when the cache is having problems (e.g. the bulk
    upload endpoint calls this when refresh_cache() itself failed), so letting
    it raise would turn an already-committed, successful write into a 500."""
    if not acronyms:
        return
    keys = [_cache_key(a.strip().upper()) for a in acronyms]
    try:
        cache_client.delete_many(keys)
    except Exception as e:
        logger.warning(f"Acronym cache invalidation failed for {len(keys)} key(s): {e}")


def refresh_cache(acronyms: List[str]) -> None:
    """Re-cache expansions for exactly the given acronyms (e.g. after a bulk
    upload), instead of re-warming the entire cache via load_acronym_cache().

    Deliberately synchronous, same reasoning as load_acronym_cache() —
    callers must run this via run_in_threadpool rather than awaiting it
    directly."""
    if not acronyms:
        return

    db = SessionLocal()
    try:
        mappings = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym.in_(acronyms), AcronymMapping.is_active.is_(True))
            .all()
        )
    finally:
        db.close()

    for mapping in mappings:
        if not _valid_expansions(mapping.expansions):
            logger.warning(
                f"Skipping cache refresh for {mapping.acronym!r}: malformed "
                f"expansions ({mapping.expansions!r})"
            )
            continue
        cache_client.set(_cache_key(mapping.acronym), json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL)


def load_acronym_cache() -> int:
    """Pre-populate the cache-aside store with every active acronym at
    startup, so first-touch queries after boot are already cache hits.
    Not a substitute for get_expansions_batch()'s DB fallback — acronyms
    added after startup, or evicted via TTL, are still served by that path.

    Deliberately synchronous — every call inside is blocking, so async
    callers must run this via run_in_threadpool or the ~600 sequential
    Redis round-trips stall the whole event loop."""
    db = SessionLocal()
    try:
        mappings = db.query(AcronymMapping).filter(AcronymMapping.is_active.is_(True)).all()
    finally:
        db.close()

    warmed = 0
    for mapping in mappings:
        if not _valid_expansions(mapping.expansions):
            logger.warning(
                f"Skipping cache warm for {mapping.acronym!r}: malformed "
                f"expansions ({mapping.expansions!r})"
            )
            continue
        try:
            cache_client.set(_cache_key(mapping.acronym), json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL)
        except Exception as e:
            logger.warning(f"Cache warm failed for {mapping.acronym!r}, skipping: {e}")
            continue
        warmed += 1

    logger.info(f"Acronym cache warmed: {warmed} active acronym(s)")
    return warmed


def _split_expansions(raw: str) -> List[str]:
    """Pipe-separated -> deduped list, order preserved (spec §5/§7).

    Deliberately duplicated from the identical helper in migration
    f13a664a31b6 rather than imported: Alembic migrations must stay
    self-contained snapshots, frozen at the point they were written, so a
    future change to this live-app helper can never silently alter what an
    already-applied historical migration does on re-run."""
    seen = set()
    result = []
    for part in raw.split("|"):
        expansion = part.strip()
        if expansion and expansion not in seen:
            seen.add(expansion)
            result.append(expansion)
    return result


# Read off the model rather than hardcoded, so this can't silently drift out
# of sync if the column length is ever changed there.
_ACRONYM_MAX_LENGTH = AcronymMapping.__table__.c.acronym.type.length


def bulk_upsert(rows: List[dict]) -> Tuple[List[str], List[str], List[str], List[dict]]:
    """Validate and upsert a batch of CSV rows (spec §7) in a single
    transaction. A row missing acronym/expansions, or an in-batch duplicate
    (Postgres' ON CONFLICT can't affect the same row twice), is recorded as
    an error and skipped rather than aborting the batch — last occurrence
    wins. Does not touch the cache — the caller owns refresh-vs-invalidate.
    Returns (created_acronyms, updated_acronyms, deactivated_acronyms, errors).
    deactivated_acronyms is a subset of created+updated (whichever rows had
    is_active=false in this batch), not a separate category.
    """
    errors: List[dict] = []
    valid_by_acronym: dict = {}

    for index, row in enumerate(rows):
        raw_acronym = (row.get(ACRONYM_CSV_COLUMN_ACRONYM) or "").strip()
        acronym = raw_acronym.upper()
        expansions = _split_expansions(row.get(ACRONYM_CSV_COLUMN_EXPANSIONS) or "")
        description = (row.get(ACRONYM_CSV_COLUMN_DESCRIPTION) or "").strip() or None
        status = (row.get(ACRONYM_CSV_COLUMN_IS_ACTIVE) or "").strip().lower()

        if not acronym or not expansions:
            errors.append({
                "index": index,
                "acronym": raw_acronym or None,
                "reason": "acronym and expansions must be non-empty after trimming whitespace",
            })
            continue

        # Optional column — missing/empty defaults to active (backward-compatible
        # with every CSV that predates this column). Only "true"/"false" are
        # accepted; anything else is a per-row error rather than a silent guess.
        if not status:
            is_active = True
        elif status == "true":
            is_active = True
        elif status == "false":
            is_active = False
        else:
            errors.append({
                "index": index,
                "acronym": acronym,
                "reason": f"is_active must be 'true' or 'false' if present, got {status!r}",
            })
            continue

        if len(acronym) > _ACRONYM_MAX_LENGTH:
            # Must be caught here, before the batch insert: an over-length
            # value reaching Postgres raises StringDataRightTruncation on the
            # single multi-row INSERT, which fails the ENTIRE batch (including
            # every otherwise-valid row) rather than just this one row.
            errors.append({
                "index": index,
                "acronym": acronym,
                "reason": f"acronym exceeds max length of {_ACRONYM_MAX_LENGTH} characters",
            })
            continue

        if acronym in valid_by_acronym:
            errors.append({
                "index": valid_by_acronym[acronym]["index"],
                "acronym": acronym,
                "reason": f"duplicate acronym in batch, superseded by row {index}",
            })

        valid_by_acronym[acronym] = {
            "index": index,
            "expansions": expansions,
            "description": description,
            "is_active": is_active,
        }

    if not valid_by_acronym:
        return [], [], [], errors

    acronym_table = sa.table(
        "acronym_mapping",
        sa.column("acronym", sa.String),
        sa.column("expansions", JSONB),
        sa.column("description", sa.Text),
        sa.column("is_active", sa.Boolean),
        sa.column("created_at", sa.DateTime),
        sa.column("updated_at", sa.DateTime),
    )
    now = datetime.now(timezone.utc)
    incoming_acronyms = list(valid_by_acronym.keys())

    db = SessionLocal()
    try:
        # Determine create vs. update (spec §7 counts them separately) before
        # the upsert — ON CONFLICT itself doesn't tell us which branch fired
        # per row.
        existing = {
            row.acronym
            for row in db.query(AcronymMapping.acronym)
            .filter(AcronymMapping.acronym.in_(incoming_acronyms))
            .all()
        }
        created = [a for a in incoming_acronyms if a not in existing]
        updated = [a for a in incoming_acronyms if a in existing]
        deactivated = [a for a in incoming_acronyms if not valid_by_acronym[a]["is_active"]]

        values = [
            {
                "acronym": acronym,
                "expansions": v["expansions"],
                "description": v["description"],
                "is_active": v["is_active"],
                "created_at": now,
                "updated_at": now,
            }
            for acronym, v in valid_by_acronym.items()
        ]
        stmt = pg_insert(acronym_table).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["acronym"],
            set_={
                "expansions": stmt.excluded.expansions,
                "description": stmt.excluded.description,
                "is_active": stmt.excluded.is_active,
                "updated_at": stmt.excluded.updated_at,
            },
        )
        db.execute(stmt)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return created, updated, deactivated, errors
