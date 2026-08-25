import redis
from redis.backoff import NoBackoff
from redis.retry import Retry

from app.config import settings

# Shared, always-on Redis connection for cache-aside style lookups (see cache_client.py).
# Independent of app.core.clients.redis_cache.redis_cache, which is the query-result LRU
# cache gated behind REDIS_CACHE_ENABLED (hardcoded False in config.py today) — that flag
# only concerns that specific feature, not Redis availability in general.
redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD or None,
    db=settings.REDIS_DB,
    decode_responses=True,
    # Without these, redis-py defaults to no timeout at all — a blackholed
    # connection would hang every read/write (and everything downstream on the
    # synchronous search path) until the OS-level TCP timeout, not the
    # exception acronym_service.get_expansions_batch()'s try/except is built to catch.
    socket_connect_timeout=settings.REDIS_SOCKET_CONNECT_TIMEOUT,
    socket_timeout=settings.REDIS_SOCKET_TIMEOUT,
    # Zero retries: redis-py's default (10 attempts, backoff) can take most of
    # a minute to raise on a blackholed connection. get_expansions_batch()
    # already falls back to Postgres on the first error, so retrying here
    # would only delay that fallback.
    retry=Retry(NoBackoff(), 0),
)
