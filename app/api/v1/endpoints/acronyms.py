import csv
import io
import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.api.deps import verify_internal_token
from app.config import settings
from app.constants import ACRONYM_CSV_COLUMN_ACRONYM, ACRONYM_CSV_COLUMN_EXPANSIONS
from app.models.api_models import AcronymBulkUploadResponse
from app.services.acronym_service import bulk_upsert, invalidate_cache, refresh_cache

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/bulk", response_model=AcronymBulkUploadResponse, dependencies=[Depends(verify_internal_token)])
async def bulk_upload_acronyms(file: UploadFile = File(...)):
    """Internal-only: upsert acronym -> expansions rows from a CSV upload (spec
    §7). Columns: acronym, expansions (pipe-separated), description (optional),
    is_active (optional, "true"/"false" — defaults to active if omitted).
    A bad row is reported in `errors`, not a batch failure — the rest commits.
    """
    raw = await file.read()
    max_bytes = settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds max size of {settings.ACRONYM_BULK_UPLOAD_MAX_SIZE_MB}MB",
        )

    try:
        # utf-8-sig strips a leading BOM if present (common in CSVs exported
        # from Excel/Sheets) and is otherwise identical to plain utf-8 — a
        # BOM left in place would silently become part of the first header
        # name ("﻿acronym"), making every row fail acronym validation.
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text))
    required_columns = {ACRONYM_CSV_COLUMN_ACRONYM, ACRONYM_CSV_COLUMN_EXPANSIONS}
    if not reader.fieldnames or not required_columns.issubset(reader.fieldnames):
        raise HTTPException(
            status_code=400,
            detail=f"CSV must have columns {sorted(required_columns)}, found {reader.fieldnames}",
        )

    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file has no data rows")

    # bulk_upsert/refresh_cache/invalidate_cache are all synchronous, blocking
    # Postgres/Redis I/O — run_in_threadpool keeps them off the event loop.
    created, updated, deactivated, errors = await run_in_threadpool(bulk_upsert, rows)

    # Spec §7: commit first (bulk_upsert already did), then refresh the cache.
    # deactivated rows go straight to invalidate_cache (no DB round-trip needed,
    # bulk_upsert already knows their status) — refresh_cache's own query filters
    # to is_active=true, so it would silently skip them and leave their old
    # cached expansion in place. If anything in this block fails, invalidate the
    # whole batch instead so the next lookup reloads from Postgres rather than
    # serving stale data.
    if created or updated:
        active_batch = [a for a in (created + updated) if a not in deactivated]
        try:
            if active_batch:
                await run_in_threadpool(refresh_cache, active_batch)
            if deactivated:
                await run_in_threadpool(invalidate_cache, deactivated)
        except Exception as e:
            logger.warning(f"Cache refresh failed after bulk upload, invalidating instead: {e}")
            await run_in_threadpool(invalidate_cache, created + updated)

    logger.info(
        f"Acronym bulk upload: {len(created)} created, {len(updated)} updated, "
        f"{len(errors)} error(s)"
    )
    return AcronymBulkUploadResponse(
        received=len(rows),
        created=len(created),
        updated=len(updated),
        errors=errors,
    )
