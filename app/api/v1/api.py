from fastapi import APIRouter
from app.api.v1.endpoints import documents, query, cache, acronyms

api_router = APIRouter()

api_router.include_router(documents.router, prefix="", tags=["documents"])
api_router.include_router(query.router, prefix="/query", tags=["query"])
api_router.include_router(cache.router, prefix="/cache", tags=["cache"])
api_router.include_router(acronyms.router, prefix="/acronyms", tags=["acronyms"])
