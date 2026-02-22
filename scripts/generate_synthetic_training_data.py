#!/usr/bin/env python3
"""
합성 학습 데이터 생성기 (Gemini Text API)

MediaPipe 연속형 변수를 기반으로 Gemini에게 추천/비추천 스타일 요청
이미지 없이 텍스트만으로 합성 데이터 생성

Author: HairMe ML Team
Date: 2025-11-15
"""

import os
import json
import time
import numpy as np
import google.generativeai as genai
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """합성 학습 데이터 생성기"""

    def __init__(self, api_key: str):
        """
        초기화

        Args:
            api_key: Gemini API 키
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")

        # 얼굴형별 연속형 변수 범위 (실제 MediaPipe 측정 기반)
        self.face_templates = {
            "긴형": {
                "face_ratio": (1.4, 1.6),
                "forehead_ratio": (0.85, 0.95),
                "jaw_ratio": (0.75, 0.85),
            },
            "둥근형": {
                "face_ratio": (0.8, 1.0),
                "forehead_ratio": (0.95, 1.05),
                "jaw_ratio": (0.90, 1.00),
            },
            "각진형": {
                "face_ratio": (1.1, 1.3),
                "forehead_ratio": (0.85, 0.95),
                "jaw_ratio": (0.85, 1.00),
            },
            "계란형": {
                "face_ratio": (1.1, 1.3),
                "forehead_ratio": (0.88, 0.98),
                "jaw_ratio": (0.80, 0.90),
            },
            "하트형": {
                "face_ratio": (1.2, 1.4),
                "forehead_ratio": (0.75, 0.85),
                "jaw_ratio": (0.65, 0.75),
            },
        }

        # 피부톤별 ITA/Hue 범위
        self.skin_templates = {
            "봄웜": {"ITA": (41, 70), "hue": (8, 14)},
            "여름쿨": {"ITA": (28, 55), "hue": (15, 25)},
            "가을웜": {"ITA": (10, 35), "hue": (5, 12)},
            "겨울쿨": {"ITA": (-10, 25), "hue": (18, 30)},
        }

    def generate_continuous_features(
        self, face_shape: str, skin_tone: str
    ) -> Dict[str, float]:
        """
        얼굴형/피부톤에 맞는 연속형 변수 샘플링

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤

        Returns:
            연속형 변수 딕셔너리
        """
        face_template = self.face_templates.get(
            face_shape, self.face_templates["계란형"]
        )
        skin_template = self.skin_templates.get(skin_tone, self.skin_templates["봄웜"])

        # 랜덤 샘플링
        face_ratio = np.random.uniform(*face_template["face_ratio"])
        forehead_ratio = np.random.uniform(*face_template["forehead_ratio"])
        jaw_ratio = np.random.uniform(*face_template["jaw_ratio"])

        ITA = np.random.uniform(*skin_template["ITA"])
        hue = np.random.uniform(*skin_template["hue"])

        # 절대값 계산 (비율에서 역산)
        cheekbone_width = np.random.uniform(200, 250)
        forehead_width = cheekbone_width * forehead_ratio
        jaw_width = cheekbone_width * jaw_ratio

        return {
            "face_ratio": round(face_ratio, 3),
            "forehead_width": round(forehead_width, 1),
            "cheekbone_width": round(cheekbone_width, 1),
            "jaw_width": round(jaw_width, 1),
            "forehead_ratio": round(forehead_ratio, 3),
            "jaw_ratio": round(jaw_ratio, 3),
            "ITA_value": round(ITA, 2),
            "hue_value": round(hue, 2),
        }

    def ask_gemini_for_recommendations(
        self, face_shape: str, skin_tone: str, features: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """
        Gemini Text API로 추천/비추천 스타일 요청

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            features: 연속형 변수

        Returns:
            {"recommended": [...], "not_recommended": [...]}
        """
        prompt = f"""당신은 전문 헤어 스타일리스트입니다. 다음 얼굴 측정 데이터를 기반으로 남성 헤어스타일을 추천해주세요.

**얼굴 측정 데이터:**
- 얼굴형: {face_shape}
- 피부톤: {skin_tone}
- 얼굴 비율(높이/너비): {features['face_ratio']:.2f}
- 이마 너비: {features['forehead_width']:.0f}px
- 광대 너비: {features['cheekbone_width']:.0f}px
- 턱 너비: {features['jaw_width']:.0f}px
- 이마/광대 비율: {features['forehead_ratio']:.2f}
- 턱/광대 비율: {features['jaw_ratio']:.2f}
- ITA 피부톤: {features['ITA_value']:.1f}°
- 색조(Hue): {features['hue_value']:.1f}

**요청사항:**
1. 이 얼굴형과 피부톤에 **가장 잘 어울리는** 남성 헤어스타일 3개 추천
2. 이 얼굴형과 피부톤에 **어울리지 않는** 남성 헤어스타일 3개 제시

**응답 형식 (JSON만):**
{{
  "recommended": [
    "스타일명1 (예: 댄디 컷)",
    "스타일명2",
    "스타일명3"
  ],
  "not_recommended": [
    "스타일명1",
    "스타일명2",
    "스타일명3"
  ],
  "reasoning": "간단한 추천 이유 (1-2줄)"
}}

**중요:**
- 한국 남성들이 실제로 미용실에서 요청하는 스타일명 사용
- 예: 댄디 컷, 투 블럭 컷, 리젠트 펌, 쉐도우 펌, 가르마 펌, 히피 펌 등"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # 다양성을 위해 0.7
                ),
            )

            raw_text = response.text.strip()

            # JSON 파싱
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            result = json.loads(raw_text.strip())

            logger.info(
                f"✅ Gemini 응답: 추천 {len(result['recommended'])}개, 비추천 {len(result['not_recommended'])}개"
            )

            return result

        except Exception as e:
            logger.error(f"❌ Gemini 요청 실패: {str(e)}")
            # 기본값 반환
            return {
                "recommended": ["댄디 컷", "투 블럭 컷", "리젠트 컷"],
                "not_recommended": ["애즈 펌", "히피 펌", "볼드 컷"],
                "reasoning": "API 오류로 기본값 사용",
            }

    def generate_training_samples(
        self, num_samples_per_combination: int = 3
    ) -> List[Dict]:
        """
        모든 얼굴형/피부톤 조합에 대해 학습 데이터 생성

        Args:
            num_samples_per_combination: 조합당 생성할 샘플 수

        Returns:
            학습 데이터 리스트
        """
        training_data = []

        face_shapes = list(self.face_templates.keys())
        skin_tones = list(self.skin_templates.keys())

        total_combinations = len(face_shapes) * len(skin_tones)
        current = 0

        for face_shape in face_shapes:
            for skin_tone in skin_tones:
                current += 1
                logger.info(
                    f"\n[{current}/{total_combinations}] {face_shape} + {skin_tone}"
                )

                for sample_idx in range(num_samples_per_combination):
                    # 1. 연속형 변수 샘플링
                    features = self.generate_continuous_features(face_shape, skin_tone)

                    # 2. Gemini에게 추천/비추천 요청
                    gemini_result = self.ask_gemini_for_recommendations(
                        face_shape, skin_tone, features
                    )

                    # 3. 추천 스타일 → 점수 부여 (순위별 차등)
                    for rank, style in enumerate(gemini_result["recommended"][:3], 1):
                        # 순위별 점수: 1위(95), 2위(85), 3위(75)
                        score = 95 - (rank - 1) * 10

                        training_data.append(
                            {
                                "face_shape": face_shape,
                                "skin_tone": skin_tone,
                                "hairstyle": style,
                                "score": score,
                                "source": "gemini_recommended",
                                "rank": rank,
                                **features,
                            }
                        )

                    # 4. 비추천 스타일 → 낮은 점수
                    for style in gemini_result["not_recommended"][:3]:
                        # 비추천 점수: 10~30 (랜덤)
                        score = np.random.uniform(10, 30)

                        training_data.append(
                            {
                                "face_shape": face_shape,
                                "skin_tone": skin_tone,
                                "hairstyle": style,
                                "score": round(score, 1),
                                "source": "gemini_not_recommended",
                                "rank": None,
                                **features,
                            }
                        )

                    logger.info(
                        f"  샘플 {sample_idx+1}: {len(gemini_result['recommended'])}개 추천, {len(gemini_result['not_recommended'])}개 비추천"
                    )

                    # API 호출 제한 (1초 대기)
                    time.sleep(1.0)

        logger.info(f"\n✅ 총 {len(training_data)}개 학습 샘플 생성 완료")

        return training_data

    def save_to_npz(self, training_data: List[Dict], output_path: str):
        """
        학습 데이터를 NPZ 형식으로 저장

        Args:
            training_data: 학습 데이터 리스트
            output_path: 출력 파일 경로
        """
        # 특징과 라벨 분리
        face_features = []
        skin_features = []
        hairstyles = []
        scores = []
        metadata = []

        for sample in training_data:
            face_features.append(
                [
                    sample["face_ratio"],
                    sample["forehead_width"],
                    sample["cheekbone_width"],
                    sample["jaw_width"],
                    sample["forehead_ratio"],
                    sample["jaw_ratio"],
                ]
            )

            skin_features.append([sample["ITA_value"], sample["hue_value"]])

            hairstyles.append(sample["hairstyle"])
            scores.append(sample["score"])

            metadata.append(
                {
                    "face_shape": sample["face_shape"],
                    "skin_tone": sample["skin_tone"],
                    "source": sample["source"],
                    "rank": sample["rank"],
                }
            )

        # NumPy 배열로 변환
        face_features = np.array(face_features, dtype=np.float32)
        skin_features = np.array(skin_features, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)

        # 저장
        np.savez_compressed(
            output_path,
            face_features=face_features,
            skin_features=skin_features,
            hairstyles=np.array(hairstyles, dtype=object),
            scores=scores,
            metadata=np.array(metadata, dtype=object),
        )

        logger.info(f"✅ 학습 데이터 저장 완료: {output_path}")
        logger.info(f"  - Face features: {face_features.shape}")
        logger.info(f"  - Skin features: {skin_features.shape}")
        logger.info(f"  - Hairstyles: {len(hairstyles)}")
        logger.info(f"  - Scores: {scores.shape}")


def main():
    """메인 실행 함수"""
    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    # 출력 디렉토리
    output_dir = Path("data_source")
    output_dir.mkdir(exist_ok=True)

    # 생성기 초기화
    generator = SyntheticDataGenerator(api_key)

    # 학습 데이터 생성
    logger.info("🚀 합성 학습 데이터 생성 시작...")
    logger.info(f"  - 얼굴형: {len(generator.face_templates)}개")
    logger.info(f"  - 피부톤: {len(generator.skin_templates)}개")
    logger.info(f"  - 조합당 샘플: 3개")
    logger.info(
        f"  - 예상 샘플 수: {len(generator.face_templates) * len(generator.skin_templates) * 3 * 6}개"
    )

    training_data = generator.generate_training_samples(num_samples_per_combination=3)

    # 저장
    output_path = output_dir / "synthetic_training_data.npz"
    generator.save_to_npz(training_data, str(output_path))

    # JSON으로도 저장 (검증용)
    json_path = output_dir / "synthetic_training_data.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(training_data[:10], f, ensure_ascii=False, indent=2)  # 처음 10개만

    logger.info(f"✅ JSON 샘플 저장: {json_path}")
    logger.info("\n🎉 완료!")


if __name__ == "__main__":
    main()
