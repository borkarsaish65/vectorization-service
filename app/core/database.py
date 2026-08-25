from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings
from app.models.db_models import Base, TranslationRecord

engine = create_engine(
    settings.DATABASE_URL,
    # Without connect_timeout, a blackholed Postgres host hangs a connection
    # attempt until the OS-level TCP timeout (minutes) — confirmed live. Every
    # caller of SessionLocal()/get_db() benefits from this, not just acronym
    # lookups (acronym_service.get_expansions_batch() also caches the
    # resulting failure so it doesn't repeat this — now-bounded — cost for
    # every candidate token in the query).
    connect_args={"connect_timeout": settings.POSTGRES_CONNECT_TIMEOUT},
)
SessionLocal = sessionmaker(bind=engine)

# Only `translations` predates Alembic and is created this way (see
# migrations/versions/fcd65d39a795_baseline_existing_translations_table.py).
# Every other table on Base (e.g. acronym_mapping) is Alembic-managed
# exclusively — scoping create_all() to just this table prevents it from
# racing Alembic's own CREATE TABLE on first boot against a fresh database.
Base.metadata.create_all(bind=engine, tables=[TranslationRecord.__table__])

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
