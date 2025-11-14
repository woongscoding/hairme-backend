"""Database schema migration utilities"""

from typing import Set, List
from sqlalchemy import text

from database.connection import SessionLocal
from core.logging import logger, log_structured


def migrate_database_schema() -> bool:
    """
    v20 schema migration (automatic execution)
    - Automatically adds missing columns if they don't exist
    - Skips if columns already exist

    Returns:
        bool: True if successful, False otherwise
    """
    if not SessionLocal:
        logger.warning("⚠️ SessionLocal is not initialized. Skipping migration.")
        return False

    try:
        db = SessionLocal()

        logger.info("🔄 DB 스키마 마이그레이션 시작...")

        # Check current table structure
        result = db.execute(text("DESCRIBE analysis_history"))
        existing_columns: Set[str] = {row[0] for row in result}

        required_columns = [
            "recommended_styles",
            "style_1_feedback",
            "style_2_feedback",
            "style_3_feedback",
            "style_1_naver_clicked",
            "style_2_naver_clicked",
            "style_3_naver_clicked",
            "feedback_at",
            # v20.1.6: vertical ratio data
            "opencv_upper_face_ratio",
            "opencv_middle_face_ratio",
            "opencv_lower_face_ratio"
        ]

        missing_columns = [col for col in required_columns if col not in existing_columns]

        if not missing_columns:
            logger.info("✅ 스키마가 이미 최신 상태입니다 (v20.1.6)")
            db.close()
            return True

        logger.info(f"🔧 누락된 컬럼 발견: {missing_columns}")

        # Prepare migration SQLs
        migration_sqls: List[str] = []

        if "recommended_styles" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN recommended_styles JSON COMMENT '추천된 3개 헤어스타일'"
            )

        if "style_1_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_1_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_2_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_2_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_3_feedback" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_3_feedback ENUM('like', 'dislike') DEFAULT NULL"
            )

        if "style_1_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_1_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "style_2_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_2_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "style_3_naver_clicked" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN style_3_naver_clicked BOOLEAN DEFAULT FALSE"
            )

        if "feedback_at" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN feedback_at DATETIME DEFAULT NULL"
            )

        # v20.1.6: vertical ratio columns
        if "opencv_upper_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_upper_face_ratio FLOAT DEFAULT NULL COMMENT '상안부 높이 비율'"
            )

        if "opencv_middle_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_middle_face_ratio FLOAT DEFAULT NULL COMMENT '중안부 높이 비율'"
            )

        if "opencv_lower_face_ratio" in missing_columns:
            migration_sqls.append(
                "ALTER TABLE analysis_history ADD COLUMN opencv_lower_face_ratio FLOAT DEFAULT NULL COMMENT '하안부 높이 비율'"
            )

        # Execute in transaction
        for sql in migration_sqls:
            logger.info(f"실행: {sql[:80]}...")
            db.execute(text(sql))

        db.commit()
        logger.info("✅ 스키마 마이그레이션 완료!")

        log_structured("schema_migration", {
            "status": "success",
            "added_columns": missing_columns
        })

        db.close()
        return True

    except Exception as e:
        logger.error(f"❌ 스키마 마이그레이션 실패: {str(e)}")
        logger.error("서버는 계속 실행되지만, v20 기능이 제대로 작동하지 않을 수 있습니다.")
        if 'db' in locals():
            db.rollback()
            db.close()
        return False
