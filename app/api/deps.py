import secrets
from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_internal_token(x_internal_token: Optional[str] = Header(None)) -> None:
    """Gate internal-only endpoints behind a shared secret (X-Internal-Token header).

    Header is optional so a missing one fails with the same 401 as a wrong one,
    not a 422 that leaks the auth mechanism. compare_digest on utf-8/
    surrogateescape-encoded bytes gives constant-time comparison (no timing
    side-channel) without crashing on non-ASCII header bytes.
    """
    if (
        not settings.INTERNAL_API_TOKEN
        or not x_internal_token
        or not secrets.compare_digest(
            x_internal_token.encode("utf-8", "surrogateescape"),
            settings.INTERNAL_API_TOKEN.encode("utf-8", "surrogateescape"),
        )
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")
