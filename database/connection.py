"""Database connection and session management"""

import urllib.parse
from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config.settings import settings
from database.models import Base
from core.logging import logger, log_structured


SessionLocal: Optional[sessionmaker] = None


def init_database() -> bool:
    """
    Initialize database connection and create tables

    Returns:
        bool: True if successful, False otherwise
    """
    global SessionLocal

    if not settings.DATABASE_URL:
        logger.warning("⚠️ DATABASE_URL이 설정되지 않았습니다.")
        return False

    try:
        logger.info(f"DATABASE_URL exists: {bool(settings.DATABASE_URL)}")
        logger.info(f"DB_PASSWORD exists: {bool(settings.DB_PASSWORD)}")
        logger.info(f"Has '://admin@' pattern: {'://admin@' in settings.DATABASE_URL}")

        # Handle password injection if needed
        if "://admin@" in settings.DATABASE_URL and settings.DB_PASSWORD:
            # URL-encode password (handle special characters)
            encoded_password = urllib.parse.quote_plus(settings.DB_PASSWORD)
            sync_db_url = settings.DATABASE_URL.replace(
                "asyncmy", "pymysql"
            ).replace(
                "://admin@",
                f"://admin:{encoded_password}@"
            )
            logger.info("Using DATABASE_URL with DB_PASSWORD injection (URL-encoded)")
        else:
            sync_db_url = settings.DATABASE_URL.replace("asyncmy", "pymysql")
            logger.info("Using DATABASE_URL as-is (already has password)")

        # Mask password for security in logs
        masked_url = sync_db_url.split('@')[0].split(':')[:-1]
        db_host = sync_db_url.split('@')[1] if '@' in sync_db_url else 'unknown'
        logger.info(f"Connecting to DB (masked): mysql+pymysql://admin:***@{db_host}")

        # Create engine
        engine = create_engine(
            sync_db_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )

        # Create session factory
        SessionLocal = sessionmaker(bind=engine)

        # Create tables
        Base.metadata.create_all(bind=engine)

        logger.info("✅ MySQL 데이터베이스 연결 성공")
        log_structured("database_connected", {
            "database": "hairme-data",
            "tables": ["analysis_history"]
        })

        return True

    except Exception as e:
        logger.error(f"❌ MySQL 연결 실패: {str(e)}")
        SessionLocal = None
        return False


def get_db() -> Session:
    """
    Get database session (FastAPI dependency)

    Yields:
        Session: SQLAlchemy database session
    """
    if SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Optional[Session]:
    """
    Get database session for direct use (not as dependency)

    Returns:
        Optional[Session]: SQLAlchemy database session or None if not initialized
    """
    if SessionLocal is None:
        return None
    return SessionLocal()
