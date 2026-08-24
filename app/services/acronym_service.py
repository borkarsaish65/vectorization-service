import json
import logging
from typing import List, Optional

from app.config import settings
from app.constants import ACRONYM_CACHE_KEY_PREFIX
from app.core.clients import cache_client
from app.core.database import SessionLocal
from app.models.db_models import AcronymMapping

logger = logging.getLogger(__name__)


def _cache_key(acronym: str) -> str:
    return f"{ACRONYM_CACHE_KEY_PREFIX}:{acronym}"


def get_expansion(acronym: str) -> Optional[List[str]]:
    """Cache-aside lookup: Redis hit returns immediately, miss falls back to
    Postgres and writes the result through to the cache for next time.

    cache_client only stores strings (Redis primitive), so the expansions
    list is JSON-encoded on write and decoded on read here rather than in
    cache_client itself, which stays acronym-agnostic.

    Normalizes to uppercase (matching how rows are stored, per spec §3/§8)
    so this is correct regardless of the caller's input case.

    Redis errors (not just misses) fall through to Postgres — a cache
    outage must degrade lookups, not break them."""
    acronym = acronym.strip().upper()
    try:
        cached = cache_client.get(_cache_key(acronym))
    except Exception as e:
        logger.warning(f"Acronym cache read failed for {acronym!r}, falling back to Postgres: {e}")
        cached = None

    if cached is not None:
        return json.loads(cached)

    db = SessionLocal()
    try:
        mapping = (
            db.query(AcronymMapping)
            .filter(AcronymMapping.acronym == acronym, AcronymMapping.is_active.is_(True))
            .first()
        )
    finally:
        db.close()

    if mapping is None:
        return None

    try:
        cache_client.set(_cache_key(acronym), json.dumps(mapping.expansions), settings.REDIS_CACHE_TTL)
    except Exception as e:
        logger.warning(f"Acronym cache write-through failed for {acronym!r}: {e}")

    return mapping.expansions


def invalidate_cache(acronym: str) -> None:
    """Drop a single acronym's cached entry. Call after any write to that row
    (e.g. the bulk upload endpoint) so stale expansions aren't served until TTL expiry."""
    cache_client.delete(_cache_key(acronym.strip().upper()))


async def load_acronym_cache() -> int:
    """Pre-populate the cache-aside store with every active acronym at startup,
    so first-touch queries after boot are already cache hits rather than DB round-trips.
    Not a substitute for get_expansion()'s per-lookup DB fallback — acronyms added
    after startup, or evicted via TTL, are still served by that path."""
    db = SessionLocal()
    try:
        rows = db.query(AcronymMapping).filter(AcronymMapping.is_active.is_(True)).all()
    finally:
        db.close()

    for row in rows:
        cache_client.set(_cache_key(row.acronym), json.dumps(row.expansions), settings.REDIS_CACHE_TTL)

    logger.info(f"Acronym cache warmed: {len(rows)} active acronym(s)")
    return len(rows)
