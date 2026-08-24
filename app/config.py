# config.py
import math
import os
from dotenv import load_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

# Disable tokenizers parallelism warning when using pdf2image for OCR
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Settings(BaseSettings):
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    # QA runs Qdrant server 1.12 while the client is pinned at 1.18 (required for BM25 sparse search).
    # The 6-minor-version gap exceeds Qdrant's allowed ≤1 diff, causing a blanket UserWarning on every
    # startup. Setting this to false suppresses that check. All operations the service uses have been
    # verified to work on server 1.12 — the warning is a false alarm for our feature set.
    # Set to true once QA server is upgraded to 1.18 to re-enable the check.
    QDRANT_CHECK_COMPATIBILITY: bool = os.getenv("QDRANT_CHECK_COMPATIBILITY", "false").lower() == "true"
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    QA_CACHE_COLLECTION: str = os.getenv("QA_CACHE_COLLECTION", "qa_cache")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    LLAMA_MODEL_ID: str = os.getenv("LLAMA_MODEL_ID", "meta.llama3-70b-instruct-v1:0")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    TRANSLATION_API_URL: str = "https://demo-api.models.ai4bharat.org/inference/translation/v2"
    PDF_CHUNK_SIZE: int = 3000
    PDF_CHUNK_OVERLAP: int = 500
    PAGE_TEXT_THRESHOLD: int = 20  # Minimum characters per page before OCR is triggered
    VECTOR_SEARCH_LIMIT: int = 1
    SIMILARITY_THRESHOLD: float = 0.40

    CHUNK_SIZE: int = 3000
    CHUNK_OVERLAP: int = 500

    # Markdown-specific settings (larger chunks to fit multiple rows for better RAG context)
    MARKDOWN_CHUNK_SIZE: int = 3500
    MARKDOWN_CHUNK_OVERLAP: int = 800
    MAX_CACHE_RESULTS: int = 1
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", 86400))  # 24 hours in seconds
    REDIS_MAX_CACHE_SIZE: int = int(os.getenv("REDIS_MAX_CACHE_SIZE", 1000))
    DATABASE_URL: str = os.getenv("POSTGRES_DATABASE_URI", "postgresql://anuj:1234@localhost:5432/ai_vector_service")
    REDIS_CACHE_ENABLED: bool = False

    # URL extraction settings
    URL_EXTRACTION_CHUNK_SIZE: int = 1500
    URL_EXTRACTION_CHUNK_OVERLAP: int = 300  # 20% overlap
    URL_REQUEST_TIMEOUT: int = 30  # seconds

    # File upload settings
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 1024))  # 1GB default (in MB)


    # Prioritized Search Configuration
    # Order determines search priority: Title > Chunk > Tags > Summary > Metadata
    SEARCH_PRIORITY_ORDER: list = ["title", "text", "tags", "summary", "metadata"]
    SEARCH_PRIORITY_WEIGHTS: dict = {
        "title": 0.36,      # 36% weight for title matches
        "text": 0.27,       # 27% weight for chunk/content matches
        "tags": 0.14,       # 14% weight for tag matches
        "summary": 0.14,    # 14% weight for summary matches
        "metadata": 0.09    # 9% weight for metadata matches
    }
    DEFAULT_SEARCH_TOP_K: int = 10
    MAX_SEARCH_TOP_K: int = 100
    MIN_SEARCH_FILTER_SCORE: int = 0
    MIN_WEIGHTED_SCORE_THRESHOLD: float = 0.0  # Minimum weighted score (15%) to include in results

    # Hybrid Search Configuration (Phase 1 — works with qdrant-client<=1.6.9)
    HYBRID_SEARCH_ENABLED: bool = os.getenv("HYBRID_SEARCH_ENABLED", "true").lower() == "true"
    EXACT_TITLE_BOOST: float = float(os.getenv("EXACT_TITLE_BOOST", "2.5"))
    PARTIAL_TITLE_BOOST: float = float(os.getenv("PARTIAL_TITLE_BOOST", "1.5"))
    # Summary boosts are lower than title (summary carries less weight than title).
    EXACT_SUMMARY_BOOST: float = float(os.getenv("EXACT_SUMMARY_BOOST", "1.4"))
    PARTIAL_SUMMARY_BOOST: float = float(os.getenv("PARTIAL_SUMMARY_BOOST", "1.2"))
    METADATA_MATCH_BOOST: float = float(os.getenv("METADATA_MATCH_BOOST", "1.2"))
    # Queries shorter than this word count skip spaCy stop-word removal
    SHORT_QUERY_THRESHOLD: int = int(os.getenv("SHORT_QUERY_THRESHOLD", "3"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))  # standard Reciprocal Rank Fusion constant

    # Sparse Vector Configuration (Phase 2 — requires qdrant-client>=1.9.0)
    SPARSE_VECTOR_NAME: str = os.getenv("SPARSE_VECTOR_NAME", "bm25")
    SPARSE_SEARCH_ENABLED: bool = os.getenv("SPARSE_SEARCH_ENABLED", "false").lower() == "true"

    # Hybrid score fusion weights. In hybrid mode the dense (cosine) and sparse
    # (BM25) scores are each min-max normalized to [0, 1] across the candidate
    # pool, then combined as: HYBRID_DENSE_WEIGHT * dense + HYBRID_SPARSE_WEIGHT * sparse.
    # This yields a calibrated 0-1 weighted_score comparable to filter_score
    # (raw RRF fused scores are ~0-0.1 and would never clear a cosine-scale
    # threshold like 0.35, silently dropping every hybrid result).
    HYBRID_DENSE_WEIGHT: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))
    HYBRID_SPARSE_WEIGHT: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.3"))

    # Hybrid fusion method — how the dense and sparse modalities are combined into
    # the final weighted_score that orders results. Selectable per deployment:
    #   "weighted" (default): HYBRID_DENSE_WEIGHT * minmax(dense) +
    #                         HYBRID_SPARSE_WEIGHT * minmax(sparse). Score-based.
    #   "rrf": Reciprocal Rank Fusion over two lists — the combined dense list
    #          (ranked by the weighted multi-field cosine sum) and the sparse list —
    #          rrf = 1/(RRF_K+dense_rank) + 1/(RRF_K+sparse_rank), then min-max
    #          normalized to [0, 1]. Rank-based; robust across retrievers.
    # In BOTH modes the dense component remains the weighted multi-field cosine sum
    # (SEARCH_PRIORITY_WEIGHTS) — only the dense+sparse fusion step differs.
    HYBRID_FUSION_METHOD: str = os.getenv("HYBRID_FUSION_METHOD", "weighted").lower()

    # Default for the per-request `include_scoring_debug` flag. When true, search
    # responses surface the hybrid fusion breakdown (keyword_score, rrf_score,
    # dense_rank, sparse_rank) on every result. A request may still override this
    # per call by sending include_scoring_debug explicitly. Keep false in
    # production (responses stay lean given the large default top_k).
    INCLUDE_SCORING_DEBUG: bool = os.getenv("INCLUDE_SCORING_DEBUG", "false").lower() == "true"

    # Candidate pool sizing for multi-field search. Each dense named-vector search
    # (title/text/tags/summary/metadata) and the sparse BM25 search retrieves up to
    # `min(top_k * SEARCH_CANDIDATE_FANOUT, SEARCH_CANDIDATE_MAX)` candidates; the
    # union is fused/ranked client-side. The CAP bounds HNSW `ef` — the dominant
    # query cost — so a large top_k cannot trigger a 10k-deep traversal per field;
    # the FANOUT gives small-top_k callers a re-ranking margin (retrieve more than
    # you return). For top_k=1000 the CAP wins → 2000/field (was 10000).
    #
    # CAP also influences result `count`: the hybrid score is min-max normalized
    # across the candidate pool, so a larger pool lets more docs clear filter_score.
    # 2000 balances latency (~2x faster than the old 10000) against count fidelity.
    # Raise toward 10000 for fuller counts, lower toward 500 for max speed.
    SEARCH_CANDIDATE_FANOUT: int = int(os.getenv("SEARCH_CANDIDATE_FANOUT", "8"))
    SEARCH_CANDIDATE_MAX: int = int(os.getenv("SEARCH_CANDIDATE_MAX", "2000"))

    # Environment configuration
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "local")  # local, production, staging, etc.

    @model_validator(mode='after')
    def _validate_fusion_config(self) -> 'Settings':
        valid_methods = {"weighted", "rrf"}
        if self.HYBRID_FUSION_METHOD not in valid_methods:
            raise ValueError(
                f"HYBRID_FUSION_METHOD must be one of {sorted(valid_methods)}, "
                f"got {self.HYBRID_FUSION_METHOD!r}"
            )
        dw, sw = self.HYBRID_DENSE_WEIGHT, self.HYBRID_SPARSE_WEIGHT
        if not (math.isfinite(dw) and dw >= 0.0):
            raise ValueError(
                f"HYBRID_DENSE_WEIGHT must be a finite non-negative number, got {dw}"
            )
        if not (math.isfinite(sw) and sw >= 0.0):
            raise ValueError(
                f"HYBRID_SPARSE_WEIGHT must be a finite non-negative number, got {sw}"
            )
        if dw + sw > 1.0 + 1e-9:
            raise ValueError(
                f"HYBRID_DENSE_WEIGHT + HYBRID_SPARSE_WEIGHT must not exceed 1.0 "
                f"(got {dw} + {sw} = {dw + sw:.6f})"
            )
        return self


settings = Settings()
