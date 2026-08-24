from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_internal_token(x_internal_token: Optional[str] = Header(None)) -> None:
    """Gate internal-only endpoints behind a shared secret (X-Internal-Token header).

    Header is optional at the FastAPI level so a missing header fails with the
    same 401 as a wrong one, instead of a 422 that would otherwise leak "a header
    is expected here" ahead of the auth check. No fallback/default token — an
    unset INTERNAL_API_TOKEN rejects every request rather than accepting an
    empty header value as valid.
    """
    if not settings.INTERNAL_API_TOKEN or x_internal_token != settings.INTERNAL_API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing internal token")
