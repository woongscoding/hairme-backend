#!/usr/bin/env python3
"""
기존 범주형 데이터를 연속형으로 근사 변환 (부정확하지만 참고용)

⚠️ 주의: 이 방법은 정확하지 않으므로 초기 학습에만 사용 권장
실제 학습에는 새로운 MediaPipe 데이터를 사용하세요

Author: HairMe ML Team
Date: 2025-11-15
"""

import numpy as np
import mysql.connector
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegacyDataConverter:
    """기존 범주형 데이터를 연속형으로 근사 변환"""

    # 얼굴형별 평균 연속형 변수 (MediaPipe 실측 데이터 기반 추정)
    FACE_SHAPE_APPROXIMATIONS = {
        "긴형": {
            "face_ratio": 1.50,
            "forehead_width": 200,
            "cheekbone_width": 220,
            "jaw_width": 175,
            "forehead_ratio": 0.91,
            "jaw_ratio": 0.80
        },
        "둥근형": {
            "face_ratio": 0.95,
            "forehead_width": 230,
            "cheekbone_width": 245,
            "jaw_width": 235,
            "forehead_ratio": 0.94,
            "jaw_ratio": 0.96
        },
        "각진형": {
            "face_ratio": 1.20,
            "forehead_width": 210,
            "cheekbone_width": 235,
            "jaw_width": 220,
            "forehead_ratio": 0.89,
            "jaw_ratio": 0.94
        },
        "계란형": {
            "face_ratio": 1.25,
            "forehead_width": 210,
            "cheekbone_width": 235,
            "jaw_width": 195,
            "forehead_ratio": 0.89,
            "jaw_ratio": 0.83
        },
        "하트형": {
            "face_ratio": 1.30,
            "forehead_width": 190,
            "cheekbone_width": 235,
            "jaw_width": 165,
            "forehead_ratio": 0.81,
            "jaw_ratio": 0.70
        }
    }

    # 피부톤별 평균 ITA/Hue 값
    SKIN_TONE_APPROXIMATIONS = {
        "봄웜": {"ITA": 50, "hue": 11},
        "여름쿨": {"ITA": 40, "hue": 18},
        "가을웜": {"ITA": 25, "hue": 9},
        "겨울쿨": {"ITA": 15, "hue": 22}
    }

    def __init__(self, db_config: Dict):
        """
        초기화

        Args:
            db_config: DB 연결 설정
        """
        self.db_config = db_config

    def approximate_features(
        self,
        face_shape: str,
        skin_tone: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        범주형 → 연속형 근사 변환 (부정확!)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            (face_features, skin_features)
        """
        # 기본값 (알 수 없는 경우)
        face_approx = self.FACE_SHAPE_APPROXIMATIONS.get(
            face_shape,
            self.FACE_SHAPE_APPROXIMATIONS["계란형"]
        )
        skin_approx = self.SKIN_TONE_APPROXIMATIONS.get(
            skin_tone,
            self.SKIN_TONE_APPROXIMATIONS["봄웜"]
        )

        # ⚠️ 노이즈 추가 (같은 카테고리도 약간씩 다르게)
        noise_scale = 0.05  # 5% 노이즈

        face_features = np.array([
            face_approx["face_ratio"] * (1 + np.random.normal(0, noise_scale)),
            face_approx["forehead_width"] * (1 + np.random.normal(0, noise_scale)),
            face_approx["cheekbone_width"] * (1 + np.random.normal(0, noise_scale)),
            face_approx["jaw_width"] * (1 + np.random.normal(0, noise_scale)),
            face_approx["forehead_ratio"] * (1 + np.random.normal(0, noise_scale)),
            face_approx["jaw_ratio"] * (1 + np.random.normal(0, noise_scale))
        ], dtype=np.float32)

        skin_features = np.array([
            skin_approx["ITA"] * (1 + np.random.normal(0, noise_scale)),
            skin_approx["hue"] * (1 + np.random.normal(0, noise_scale))
        ], dtype=np.float32)

        return face_features, skin_features

    def convert_legacy_data(self) -> Dict:
        """
        기존 DB 데이터를 연속형으로 근사 변환

        Returns:
            {"face_features": [...], "skin_features": [...], ...}
        """
        logger.info("🔄 기존 데이터 변환 시작...")

        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor(dictionary=True)

        # 피드백이 있는 레코드만 조회
        query = """
        SELECT
            id,
            face_shape,
            personal_color,
            JSON_EXTRACT(recommended_styles, '$[0].style_name') as style_1,
            JSON_EXTRACT(recommended_styles, '$[1].style_name') as style_2,
            JSON_EXTRACT(recommended_styles, '$[2].style_name') as style_3,
            style_1_feedback,
            style_2_feedback,
            style_3_feedback,
            style_1_naver_clicked,
            style_2_naver_clicked,
            style_3_naver_clicked
        FROM analysis_history
        WHERE (style_1_feedback IS NOT NULL
            OR style_2_feedback IS NOT NULL
            OR style_3_feedback IS NOT NULL)
        """

        cursor.execute(query)
        records = cursor.fetchall()

        logger.info(f"📊 변환 대상: {len(records)}개 레코드")

        converted_data = []

        for record in records:
            face_shape = record["face_shape"]
            skin_tone = record["personal_color"]

            # 연속형 근사
            face_feat, skin_feat = self.approximate_features(face_shape, skin_tone)

            # 각 스타일별 피드백 처리
            for idx in range(1, 4):
                style = record.get(f"style_{idx}")
                feedback = record.get(f"style_{idx}_feedback")
                naver_clicked = record.get(f"style_{idx}_naver_clicked", False)

                if not style or not feedback:
                    continue

                # 점수 매핑
                if feedback == "like":
                    score = 90
                    if naver_clicked:
                        score = 95
                elif feedback == "dislike":
                    score = 20
                else:
                    continue

                # 스타일명 정리 (JSON 따옴표 제거)
                style = style.strip('"')

                converted_data.append({
                    "face_features": face_feat.copy(),
                    "skin_features": skin_feat.copy(),
                    "hairstyle": style,
                    "score": score,
                    "source": "legacy_approximated",
                    "original_id": record["id"]
                })

        cursor.close()
        conn.close()

        logger.info(f"✅ 변환 완료: {len(converted_data)}개 샘플")
        logger.warning("⚠️ 주의: 이 데이터는 근사치이므로 정확도가 낮습니다!")

        return {
            "samples": converted_data,
            "total_count": len(converted_data),
            "source": "legacy_approximated"
        }


# ========== 사용 예시 ==========
if __name__ == "__main__":
    db_config = {
        "host": "hairme-data.cr28a6uqo2k8.ap-northeast-2.rds.amazonaws.com",
        "user": "admin",
        "password": "Hairstyle!2580",
        "database": "hairme"
    }

    converter = LegacyDataConverter(db_config)
    result = converter.convert_legacy_data()

    # NPZ 저장
    import numpy as np

    samples = result["samples"]
    np.savez_compressed(
        "data_source/legacy_approximated_data.npz",
        face_features=np.array([s["face_features"] for s in samples]),
        skin_features=np.array([s["skin_features"] for s in samples]),
        hairstyles=np.array([s["hairstyle"] for s in samples]),
        scores=np.array([s["score"] for s in samples])
    )

    logger.info("✅ 저장 완료: data_source/legacy_approximated_data.npz")
    logger.warning("⚠️ 이 데이터는 보조 학습용으로만 사용하세요!")
