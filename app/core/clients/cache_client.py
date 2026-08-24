from typing import Optional

from app.core.clients.redis_client import redis_client


def get(key: str) -> Optional[str]:
    return redis_client.get(key)


def set(key: str, value: str, ttl: int) -> None:
    redis_client.setex(key, ttl, value)


def delete(key: str) -> None:
    redis_client.delete(key)
