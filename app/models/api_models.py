from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from app.config import settings

class DocumentMetadata(BaseModel):
    source: str
    page: Optional[int] = None
    row: Optional[int] = None

class DocumentUploadRequest(BaseModel):
    priority: str

class QueryRequest(BaseModel):
    query: str
    search_limit: Optional[int] = settings.VECTOR_SEARCH_LIMIT
    force_llm: Optional[bool] = False

class DeleteRequest(BaseModel):
    source_id: str = Field(..., description="Unique source identifier to delete documents")
    company_id: Optional[str] = Field(None, description="Company ID to filter deletion")

class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, Any]
    score: float

class MultilingualQueryResponse(BaseModel):
    relevant_texts: List[Dict[str, Any]]
    original_query: str
    translated_query: Optional[str] = None
    language: str

class MultilingualQueryRequest(BaseModel):
    query: str
    search_limit: int = settings.VECTOR_SEARCH_LIMIT
    priority_filter: Optional[str] = None

class SimilarityCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to check for similarity")
    company_id: str = Field(..., description="Company ID to filter by")
    threshold: float = Field(default=0.85, description="Similarity threshold")
    exclude_source_id: Optional[str] = Field(None, description="Source ID to exclude")

class SimilarityCheckResponse(BaseModel):
    has_similar: bool
    similar_documents: List[Dict[str, Any]]

class DetailFilterScore(BaseModel):
    """Field-level threshold configuration for granular filtering"""
    title: float = Field(default=0.36, ge=0.0, le=1.0, description="Minimum score threshold for title field")
    text: float = Field(default=0.27, ge=0.0, le=1.0, description="Minimum score threshold for text field")
    tags: float = Field(default=0.14, ge=0.0, le=1.0, description="Minimum score threshold for tags field")
    summary: float = Field(default=0.14, ge=0.0, le=1.0, description="Minimum score threshold for summary field")
    metadata: float = Field(default=0.09, ge=0.0, le=1.0, description="Minimum score threshold for metadata field")

class PrioritizedSearchRequest(BaseModel):
    query: Optional[str] = Field(default=None, description="Search query text (optional - if not provided, returns all documents with unique source_id)")
    top_k: int = Field(default=1000000, description="Number of top results to return")
    filter_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Minimum weighted score threshold (0–1). "
                    "Results below this score will be filtered out. "
                    "Used when detail_filter_score is None."
    )
    detail_filter_score: Optional[DetailFilterScore] = Field(
        default=None,
        description="Field-level score thresholds. When filter_score=0, this enables granular filtering. "
                    "A document passes if ANY field meets its threshold (OR logic)."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="Optional list of tag values to filter (searches in 'tags' field, OR condition)"
    )
    organizations: Optional[List[str]] = Field(
        default=None,
        description="Optional list of company names to filter (searches in 'metadata.company' field, OR condition)"
    )
    resource_type: Optional[List[str]] = Field(
        default=None,
        description="Optional list of key entities to filter (searches in 'metadata.KEY ENTITIES' field, OR condition)"
    )
    file_type: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document types to filter (searches in 'metadata.type' field, OR condition)"
    )
    search_mode: Literal["hybrid", "semantic"] = Field(
        default="hybrid",
        description="Search mode: 'hybrid' (semantic + title/summary boost) or "
                    "'semantic' (vector only, no boosts). Defaults to 'hybrid'. "
                    "Any other value is rejected with a 422 validation error."
    )
    include_scoring_debug: bool = Field(
        default=settings.INCLUDE_SCORING_DEBUG,
        description="When true, each result includes the hybrid fusion breakdown "
                    "(keyword_score, rrf_score, dense_rank, sparse_rank) for inspecting "
                    "how the final score was produced. Defaults to the INCLUDE_SCORING_DEBUG "
                    "env setting; a request may override it per call. Off by default to keep "
                    "responses lean."
    )

class SearchResultItem(BaseModel):
    id: str
    text: str
    title: Optional[str] = None
    summary: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Dict[str, Any]
    source_id: str
    score: float
    field_scores: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description=(
            "Per-field cosine similarity scores for semantically retrieved documents "
            "(keys: title, tags, summary, metadata, text, values 0–1). "
            "Internal fusion keys (rrf, bm25_sparse) are excluded. "
            "Keyword-injected documents that bypassed vector scoring carry None "
            "for unscored fields to distinguish them from a genuine zero-score."
        )
    )
    keyword_score: Optional[float] = Field(
        default=None,
        description="Raw BM25 sparse vector score (Phase 2). Surfaced BY DEFAULT whenever "
                    "hybrid/sparse search is enabled and produced a score for the document "
                    "(not gated by include_scoring_debug, unlike the other breakdown fields). "
                    "None when sparse search is disabled (dense-only mode)."
    )
    rrf_score: Optional[float] = Field(
        default=None,
        description="Raw Reciprocal Rank Fusion value before min-max normalization "
                    "(1/(k+dense_rank) + 1/(k+sparse_rank)). Populated only when "
                    "include_scoring_debug=true and HYBRID_FUSION_METHOD=rrf."
    )
    dense_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank of this document in the combined dense list (ranked by the "
                    "weighted multi-field cosine sum). Debug-only (include_scoring_debug=true)."
    )
    sparse_rank: Optional[int] = Field(
        default=None,
        description="1-indexed rank of this document in the BM25 sparse list, or None if it had no "
                    "sparse hit. Debug-only (include_scoring_debug=true)."
    )
    raw_dense: Optional[float] = Field(
        default=None,
        description="Weighted multi-field cosine sum (Σ field_weight × field_score), before any "
                    "hybrid fusion or boosting. In dense-only mode this equals the pre-boost score. "
                    "Debug-only (include_scoring_debug=true)."
    )
    normalized_dense: Optional[float] = Field(
        default=None,
        description="raw_dense min-max normalized to 0–1 across the candidate pool "
                    "((raw_dense − dense_min)/(dense_max − dense_min); see search_config.scoring_context). "
                    "Batch-relative — shifts with the query's candidate pool. None in dense-only mode. "
                    "Debug-only (include_scoring_debug=true)."
    )
    normalized_sparse: Optional[float] = Field(
        default=None,
        description="keyword_score (BM25) min-max normalized to 0–1 across the candidate pool. "
                    "Batch-relative. None when sparse search is disabled or the doc had no sparse hit. "
                    "Debug-only (include_scoring_debug=true)."
    )
    title_multiplier: Optional[float] = Field(
        default=None,
        description="Boost factor applied for a title keyword match (exact/partial). "
                    "1.0 means NO boost was applied (neutral no-op), not a phantom boost. "
                    "Debug-only (include_scoring_debug=true)."
    )
    summary_multiplier: Optional[float] = Field(
        default=None,
        description="Boost factor applied for a summary keyword match (exact/partial). "
                    "1.0 means NO boost was applied (neutral no-op), not a phantom boost. "
                    "Applied after title_multiplier, each capped at 1.0. "
                    "Debug-only (include_scoring_debug=true)."
    )
    title_match: Optional[str] = Field(
        default=None,
        description="Title match type: 'exact', 'partial', or None if no title match"
    )
    summary_match: Optional[str] = Field(
        default=None,
        description="Summary match type: 'exact', 'partial', or None if no summary match"
    )

class PrioritizedSearchResponse(BaseModel):
    query: Optional[str] = None
    total_results: int
    top_k: int
    results: List[SearchResultItem]
    search_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration used for this search"
    )
    acronym_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detected acronym(s) and their expansions, if any were found in the query"
    )

class TextSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query text")
    top_k: int = Field(default=10, description="Number of unique documents to return (default: 5)")
    threshold: float = Field(default=0.40, description="Minimum similarity score threshold (default: 0.40)")

class TextSearchResultItem(BaseModel):
    source_id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class TextSearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[TextSearchResultItem]

class SourceVerificationRequest(BaseModel):
    source_ids: List[str] = Field(..., description="List of source IDs to verify")

class SourceVerificationResponse(BaseModel):
    total_requested: int = Field(..., description="Total number of source IDs requested")
    found: List[str] = Field(..., description="List of source IDs that exist in Qdrant")
    not_found: List[str] = Field(..., description="List of source IDs that don't exist in Qdrant")
    found_count: int = Field(..., description="Count of found source IDs")
    not_found_count: int = Field(..., description="Count of not found source IDs")

class AcronymUploadError(BaseModel):
    index: int = Field(..., description="0-based index of the row in the CSV (0 = first data row, header excluded)")
    acronym: Optional[str] = Field(None, description="Acronym value, if present on the row")
    reason: str

class AcronymBulkUploadResponse(BaseModel):
    received: int
    created: int
    updated: int
    errors: List[AcronymUploadError]
