# database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text
import os
from urllib.parse import quote_plus

# 환경 변수에서 개별적으로 읽기
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "hairme")  # 기본값 'hairme'

# 비밀번호 URL 인코딩 (특수문자 처리)
DB_PASSWORD_ENCODED = quote_plus(DB_PASSWORD) if DB_PASSWORD else ""

# DB_NAME이 있으면 포함, 없으면 제외
if DB_NAME:
    DATABASE_URL = f"mysql+asyncmy://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    print(f"🔗 Connecting to database: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
else:
    DATABASE_URL = f"mysql+asyncmy://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/"
    print(f"🔗 Connecting to database: {DB_USER}@{DB_HOST}:{DB_PORT} (no database specified)")

# 비동기 엔진 생성
engine = create_async_engine(DATABASE_URL, echo=True, pool_pre_ping=True)

# 비동기 세션 생성
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

# ORM 모델의 기본이 될 Base 클래스
Base = declarative_base()


# FastAPI에서 DB 세션을 쉽게 사용하기 위한 함수
async def get_db_session():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# DB 생성 함수 (DB가 없으면 생성)
async def create_database_if_not_exists():
    """데이터베이스가 없으면 생성"""
    if not DB_NAME:
        return

    # DB 이름 없이 연결
    temp_url = f"mysql+asyncmy://{DB_USER}:{DB_PASSWORD_ENCODED}@{DB_HOST}:{DB_PORT}/"
    temp_engine = create_async_engine(temp_url, echo=False, pool_pre_ping=True)

    try:
        async with temp_engine.connect() as conn:
            # 데이터베이스 존재 확인
            result = await conn.execute(
                text(f"SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = '{DB_NAME}'")
            )
            exists = result.fetchone()

            if not exists:
                # 데이터베이스 생성
                await conn.execute(text(f"CREATE DATABASE {DB_NAME} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                await conn.commit()
                print(f"✅ Database '{DB_NAME}' created successfully")
            else:
                print(f"✅ Database '{DB_NAME}' already exists")
    except Exception as e:
        print(f"⚠️ Database creation check failed: {e}")
    finally:
        await temp_engine.dispose()