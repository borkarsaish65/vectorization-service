import redis

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
)
