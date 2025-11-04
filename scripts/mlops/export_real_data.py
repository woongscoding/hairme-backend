"""
실제 사용자 데이터를 DB에서 추출하여 학습 데이터로 변환

DB의 analysis_history 테이블에서 피드백이 있는 데이터를 추출하고,
합성 데이터와 동일한 형식의 CSV로 저장합니다.
"""

import os
import sys
import pymysql
import pandas as pd
import json
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class RealDataExporter:
    """실제 사용자 데이터 추출기"""

    def __init__(self):
        """환경변수에서 DB 연결 정보 로드"""
        self.database_url = os.getenv("DATABASE_URL")
        self.db_password = os.getenv("DB_PASSWORD")

        if not self.database_url or not self.db_password:
            raise ValueError(
                "환경변수 DATABASE_URL과 DB_PASSWORD가 설정되어 있어야 합니다.\n"
                "예: export DATABASE_URL='asyncmy://admin@hairme-data.xxx.rds.amazonaws.com:3306/hairme'"
            )

        # asyncmy -> pymysql로 변경
        self.sync_db_url = self.database_url.replace("asyncmy", "pymysql")

        # URL 파싱
        self._parse_db_url()

    def _parse_db_url(self):
        """DB URL에서 호스트, 포트, 데이터베이스 추출"""
        # pymysql://admin@host:port/dbname 형식
        url = self.sync_db_url

        # 프로토콜 제거
        url = url.replace("pymysql://", "")

        # admin@ 부분과 나머지 분리
        parts = url.split("@")
        self.user = parts[0]  # admin

        # host:port/dbname 분리
        rest = parts[1]
        host_port, dbname = rest.split("/")

        if ":" in host_port:
            self.host, port_str = host_port.split(":")
            self.port = int(port_str)
        else:
            self.host = host_port
            self.port = 3306

        self.database = dbname

        print(f"✅ DB 연결 정보:")
        print(f"   호스트: {self.host}")
        print(f"   포트: {self.port}")
        print(f"   데이터베이스: {self.database}")
        print(f"   사용자: {self.user}")

    def connect(self):
        """DB 연결"""
        try:
            self.conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.db_password,
                database=self.database,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
            print(f"✅ DB 연결 성공: {self.database}")
            return self.conn
        except Exception as e:
            print(f"❌ DB 연결 실패: {e}")
            raise

    def export_feedback_data(self):
        """피드백이 있는 데이터 추출"""

        query = """
        SELECT
            id,
            face_shape,
            personal_color,
            recommended_styles,
            style_1_feedback,
            style_2_feedback,
            style_3_feedback,
            style_1_naver_clicked,
            style_2_naver_clicked,
            style_3_naver_clicked,
            created_at,
            feedback_at
        FROM analysis_history
        WHERE
            (style_1_feedback IS NOT NULL
             OR style_2_feedback IS NOT NULL
             OR style_3_feedback IS NOT NULL)
        ORDER BY created_at DESC
        """

        with self.conn.cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()

        print(f"✅ 피드백 데이터 {len(results)}건 조회 완료")
        return results

    def transform_to_training_format(self, db_records):
        """
        DB 레코드를 학습 데이터 형식으로 변환

        합성 데이터 형식:
        face_shape, skin_tone, hairstyle, score, feedback, naver_clicked, reason
        """
        training_data = []

        for record in db_records:
            # recommended_styles JSON 파싱
            try:
                styles_json = record['recommended_styles']
                if isinstance(styles_json, str):
                    styles = json.loads(styles_json)
                else:
                    styles = styles_json
            except:
                print(f"⚠️ 레코드 {record['id']}: recommended_styles 파싱 실패")
                continue

            # 3개의 추천 스타일 각각 처리
            for i in range(1, 4):
                feedback = record.get(f'style_{i}_feedback')

                # 피드백이 없으면 스킵
                if not feedback:
                    continue

                # 해당 스타일 정보 가져오기
                style_info = styles[i-1] if len(styles) >= i else None
                if not style_info:
                    continue

                # 헤어스타일 이름
                hairstyle = style_info.get('style', 'Unknown')

                # ML 점수 (없으면 기본값)
                score = style_info.get('ml_confidence_score', 0.85)

                # 네이버 클릭 여부
                naver_clicked = record.get(f'style_{i}_naver_clicked', False)

                # 추천 이유
                reason = style_info.get('reason', '추천 이유 없음')

                # skin_tone 변환 (personal_color → skin_tone)
                skin_tone = self._convert_personal_color(record['personal_color'])

                # 학습 데이터 레코드 생성
                training_record = {
                    'face_shape': record['face_shape'],
                    'skin_tone': skin_tone,
                    'hairstyle': hairstyle,
                    'score': score,
                    'feedback': feedback,
                    'naver_clicked': naver_clicked,
                    'reason': reason
                }

                training_data.append(training_record)

        print(f"✅ 학습 데이터 {len(training_data)}건 생성 완료")
        return training_data

    def _convert_personal_color(self, personal_color):
        """
        personal_color를 skin_tone으로 변환

        DB: 봄웜톤, 여름쿨톤, 가을웜톤, 겨울쿨톤, etc
        → 학습 데이터: 웜톤, 쿨톤, 중간톤
        """
        if not personal_color:
            return "중간톤"

        personal_color = personal_color.lower()

        if "웜" in personal_color or "봄" in personal_color or "가을" in personal_color:
            return "웜톤"
        elif "쿨" in personal_color or "여름" in personal_color or "겨울" in personal_color:
            return "쿨톤"
        else:
            return "중간톤"

    def save_to_csv(self, data, output_path):
        """CSV로 저장"""
        df = pd.DataFrame(data)

        # 컬럼 순서 맞추기 (합성 데이터와 동일)
        columns_order = ['face_shape', 'skin_tone', 'hairstyle', 'score', 'feedback', 'naver_clicked', 'reason']
        df = df[columns_order]

        # CSV 저장
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 저장 완료: {output_path}")
        print(f"   총 레코드: {len(df)}건")

        # 통계 출력
        self._print_statistics(df)

    def _print_statistics(self, df):
        """데이터 통계 출력"""
        print("\n📊 데이터 통계:")
        print(f"   총 레코드: {len(df)}건")
        print(f"\n   얼굴형 분포:")
        print(df['face_shape'].value_counts().to_string())
        print(f"\n   피부톤 분포:")
        print(df['skin_tone'].value_counts().to_string())
        print(f"\n   헤어스타일 분포:")
        print(df['hairstyle'].value_counts().to_string())
        print(f"\n   피드백 분포:")
        print(df['feedback'].value_counts().to_string())
        print(f"\n   네이버 클릭률: {df['naver_clicked'].mean()*100:.1f}%")
        print(f"   평균 점수: {df['score'].mean():.3f}")

    def export(self, output_dir="data_source"):
        """전체 추출 프로세스 실행"""
        print("=" * 60)
        print("🚀 실제 사용자 데이터 추출 시작")
        print("=" * 60)

        # DB 연결
        self.connect()

        try:
            # 데이터 추출
            db_records = self.export_feedback_data()

            if len(db_records) == 0:
                print("⚠️ 피드백 데이터가 없습니다. 추출을 종료합니다.")
                return None

            # 학습 데이터 형식으로 변환
            training_data = self.transform_to_training_format(db_records)

            if len(training_data) == 0:
                print("⚠️ 변환된 학습 데이터가 없습니다.")
                return None

            # CSV 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"real_user_data_{timestamp}.csv"
            output_path = Path(project_root) / output_dir / output_filename

            # 최신 파일 링크도 생성
            latest_path = Path(project_root) / output_dir / "real_user_data_latest.csv"

            self.save_to_csv(training_data, output_path)
            self.save_to_csv(training_data, latest_path)

            print("\n" + "=" * 60)
            print("✅ 데이터 추출 완료!")
            print("=" * 60)

            return output_path

        finally:
            self.conn.close()
            print("🔌 DB 연결 종료")


def main():
    """메인 함수"""
    try:
        exporter = RealDataExporter()
        output_path = exporter.export()

        if output_path:
            print(f"\n✅ 성공: {output_path}")
            return 0
        else:
            print("\n⚠️ 데이터가 없어 파일이 생성되지 않았습니다.")
            return 1

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
