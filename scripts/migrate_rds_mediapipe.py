#!/usr/bin/env python3
"""
RDS MediaPipe 마이그레이션 실행 스크립트

db_schema_mediapipe_continuous.sql을 RDS에 안전하게 적용합니다.

✅ 체크리스트:
1. RDS 백업 확인
2. DATABASE_URL / DB_PASSWORD 확인
3. SQL 스크립트 실행
4. 컬럼 추가 확인
5. 인덱스 확인
6. 통계 확인

Author: HairMe ML Team
Date: 2025-11-15
"""

import os
import sys
import subprocess
import pymysql
import logging
from pathlib import Path
from typing import Dict, Optional
import urllib.parse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class RDSMigrator:
    """RDS 마이그레이션 실행기"""

    def __init__(self):
        """초기화"""
        self.connection = None
        self.db_config = {}

    def load_credentials(self) -> Dict:
        """AWS Secrets Manager에서 DB 정보 가져오기"""
        logger.info("📂 AWS Secrets Manager에서 DB 정보 가져오기...")

        try:
            # DATABASE_URL 가져오기
            result = subprocess.run(
                [
                    "aws", "secretsmanager", "get-secret-value",
                    "--secret-id", "hairme-database-url",
                    "--query", "SecretString",
                    "--output", "text"
                ],
                capture_output=True,
                text=True,
                env={"MSYS_NO_PATHCONV": "1", **os.environ}
            )

            database_url = result.stdout.strip()

            # DB_PASSWORD 가져오기
            result = subprocess.run(
                [
                    "aws", "secretsmanager", "get-secret-value",
                    "--secret-id", "hairme-db-password",
                    "--query", "SecretString",
                    "--output", "text"
                ],
                capture_output=True,
                text=True,
                env={"MSYS_NO_PATHCONV": "1", **os.environ}
            )

            db_password = result.stdout.strip()

            # URL 파싱
            # mysql+asyncmy://admin@hairme-data-v2.cr28a6uqo2k8.ap-northeast-2.rds.amazonaws.com:3306/hairme
            parts = database_url.split('@')[1].split('/')
            host_port = parts[0]
            database = parts[1] if len(parts) > 1 else 'hairme'

            host = host_port.split(':')[0]
            port = int(host_port.split(':')[1]) if ':' in host_port else 3306

            self.db_config = {
                'host': host,
                'port': port,
                'user': 'admin',
                'password': db_password,
                'database': database
            }

            logger.info(f"✅ DB 정보 로드 완료:")
            logger.info(f"   Host: {host}")
            logger.info(f"   Port: {port}")
            logger.info(f"   Database: {database}")

            return self.db_config

        except Exception as e:
            logger.error(f"❌ DB 정보 로드 실패: {str(e)}")
            raise

    def connect(self) -> pymysql.Connection:
        """RDS 접속"""
        logger.info("\n🔌 RDS 접속 중...")

        try:
            self.connection = pymysql.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                database=self.db_config['database'],
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )

            logger.info("✅ RDS 접속 성공")
            return self.connection

        except Exception as e:
            logger.error(f"❌ RDS 접속 실패: {str(e)}")
            raise

    def check_backup(self) -> bool:
        """백업 테이블 존재 확인"""
        logger.info("\n📦 백업 테이블 확인 중...")

        with self.connection.cursor() as cursor:
            cursor.execute("SHOW TABLES LIKE 'analysis_history_backup_mediapipe'")
            result = cursor.fetchone()

            if result:
                cursor.execute("SELECT COUNT(*) as count FROM analysis_history_backup_mediapipe")
                count = cursor.fetchone()['count']
                logger.info(f"✅ 백업 테이블 존재: {count:,}개 레코드")
                return True
            else:
                logger.warning("⚠️ 백업 테이블 없음")
                return False

    def execute_sql_file(self, sql_file_path: Path):
        """SQL 파일 실행"""
        logger.info(f"\n🔄 SQL 스크립트 실행: {sql_file_path}")

        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        # SQL 문장 분리 (세미콜론 기준)
        statements = sql_script.split(';')

        with self.connection.cursor() as cursor:
            for i, statement in enumerate(statements):
                statement = statement.strip()

                if not statement or statement.startswith('--') or statement.startswith('/*'):
                    continue

                try:
                    cursor.execute(statement)
                    self.connection.commit()

                    # 결과가 있는 경우 출력 (SELECT 문)
                    if statement.strip().upper().startswith('SELECT'):
                        results = cursor.fetchall()
                        if results:
                            logger.info(f"  Query {i+1}: {results[0]}")

                except Exception as e:
                    # 이미 존재하는 컬럼 추가 시도 등은 무시
                    if "Duplicate column name" in str(e):
                        logger.info(f"  ⚠️ 컬럼이 이미 존재함 (스킵)")
                    else:
                        logger.error(f"  ❌ 실행 실패: {statement[:100]}...")
                        logger.error(f"     오류: {str(e)}")

        logger.info("✅ SQL 스크립트 실행 완료")

    def verify_columns(self) -> bool:
        """MediaPipe 컬럼 추가 확인"""
        logger.info("\n🔍 MediaPipe 컬럼 확인 중...")

        with self.connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    COLUMN_NAME,
                    COLUMN_TYPE,
                    COLUMN_COMMENT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = %s
                  AND TABLE_NAME = 'analysis_history'
                  AND COLUMN_NAME LIKE 'mediapipe%%'
                ORDER BY ORDINAL_POSITION
            """, (self.db_config['database'],))

            columns = cursor.fetchall()

            if columns:
                logger.info(f"✅ MediaPipe 컬럼 확인됨 ({len(columns)}개):")
                for col in columns:
                    logger.info(f"   - {col['COLUMN_NAME']:30s} {col['COLUMN_TYPE']:15s} {col['COLUMN_COMMENT']}")
                return True
            else:
                logger.error("❌ MediaPipe 컬럼이 없습니다!")
                return False

    def verify_indexes(self) -> bool:
        """인덱스 확인"""
        logger.info("\n🔍 인덱스 확인 중...")

        with self.connection.cursor() as cursor:
            cursor.execute("""
                SHOW INDEXES FROM analysis_history
                WHERE Key_name IN ('idx_mediapipe_complete', 'idx_training_data')
            """)

            indexes = cursor.fetchall()

            if indexes:
                logger.info(f"✅ 인덱스 확인됨 ({len(indexes)}개):")
                for idx in indexes:
                    logger.info(f"   - {idx['Key_name']}")
                return True
            else:
                logger.warning("⚠️ 인덱스가 생성되지 않았습니다.")
                return False

    def show_statistics(self):
        """데이터 통계 확인"""
        logger.info("\n📊 데이터 통계:")

        with self.connection.cursor() as cursor:
            # 전체 레코드 수
            cursor.execute("SELECT COUNT(*) as total FROM analysis_history")
            total = cursor.fetchone()['total']

            # MediaPipe 데이터 수집률
            cursor.execute("""
                SELECT
                    COUNT(*) as total_records,
                    SUM(CASE WHEN mediapipe_features_complete = TRUE THEN 1 ELSE 0 END) as mediapipe_complete,
                    ROUND(
                        SUM(CASE WHEN mediapipe_features_complete = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
                        2
                    ) as mediapipe_coverage_percent
                FROM analysis_history
            """)

            stats = cursor.fetchone()

            logger.info(f"   - 전체 레코드: {total:,}개")
            logger.info(f"   - MediaPipe 완료: {stats['mediapipe_complete']:,}개 ({stats['mediapipe_coverage_percent']}%)")

            # 최근 10개 레코드
            cursor.execute("""
                SELECT
                    id,
                    face_shape,
                    personal_color,
                    mediapipe_face_ratio,
                    mediapipe_ITA_value,
                    mediapipe_features_complete,
                    created_at
                FROM analysis_history
                ORDER BY id DESC
                LIMIT 5
            """)

            recent = cursor.fetchall()

            logger.info(f"\n   최근 5개 레코드:")
            for row in recent:
                mp_status = "✓" if row['mediapipe_features_complete'] else "✗"
                logger.info(f"      ID {row['id']:5d}: {row['face_shape']:10s} / {row['personal_color']:10s} "
                          f"MediaPipe:{mp_status}")

    def close(self):
        """연결 종료"""
        if self.connection:
            self.connection.close()
            logger.info("\n🔌 RDS 연결 종료")


def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("🚀 RDS MediaPipe 마이그레이션 시작")
    logger.info("=" * 60)

    migrator = RDSMigrator()

    try:
        # 1. DB 정보 로드
        migrator.load_credentials()

        # 2. RDS 접속
        migrator.connect()

        # 3. 백업 확인
        backup_exists = migrator.check_backup()
        if not backup_exists:
            logger.warning("\n⚠️ 백업 테이블이 없습니다.")
            response = input("계속 진행하시겠습니까? (y/N): ")
            if response.lower() != 'y':
                logger.info("마이그레이션 취소됨")
                return

        # 4. SQL 스크립트 실행
        sql_file = PROJECT_ROOT / "db_schema_mediapipe_continuous.sql"
        if not sql_file.exists():
            logger.error(f"❌ SQL 파일이 없습니다: {sql_file}")
            return

        migrator.execute_sql_file(sql_file)

        # 5. 검증
        columns_ok = migrator.verify_columns()
        indexes_ok = migrator.verify_indexes()

        # 6. 통계
        migrator.show_statistics()

        # 결과
        logger.info("\n" + "=" * 60)
        if columns_ok:
            logger.info("✅ 마이그레이션 성공!")
        else:
            logger.error("❌ 마이그레이션 실패!")

        logger.info("=" * 60)

        logger.info("\n다음 단계:")
        logger.info("  1. 서버 코드 업데이트 (MediaPipe 연속형 변수 저장)")
        logger.info("  2. 새 데이터 수집 (mediapipe_features_complete = TRUE)")
        logger.info("  3. v4 모델 학습")
        logger.info("  4. 배포")

    except Exception as e:
        logger.error(f"\n❌ 마이그레이션 실패: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        migrator.close()


if __name__ == "__main__":
    main()
