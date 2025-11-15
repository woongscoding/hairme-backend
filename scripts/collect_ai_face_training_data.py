#!/usr/bin/env python3
"""
ThisPersonDoesNotExist.com AI 얼굴로 학습 데이터 자동 수집

🎯 프로세스:
1. ThisPersonDoesNotExist.com에서 AI 생성 얼굴 이미지 다운로드
2. MediaPipe로 연속형 변수 측정 (정확한 실측!)
3. Gemini Vision API로 헤어스타일 추천 받기
4. 학습 데이터로 저장 (NPZ + JSON)

💡 장점:
- 실제 얼굴 측정 (합성 데이터보다 정확)
- 저작권 문제 없음 (AI 생성)
- 개인정보 문제 없음 (실존 인물 아님)
- 무한 다양성

Author: HairMe ML Team
Date: 2025-11-15
"""

import os
import sys
import time
import json
import requests
import numpy as np
import google.generativeai as genai
from pathlib import Path
from typing import Optional, Dict, List
import logging
from PIL import Image
import io

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.mediapipe_analyzer import MediaPipeFaceAnalyzer, MediaPipeFaceFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


class AIFaceDataCollector:
    """AI 생성 얼굴로 학습 데이터 수집"""

    THISPERSONDOESNOTEXIST_URL = "https://thispersondoesnotexist.com/"

    def __init__(self, gemini_api_key: str):
        """
        초기화

        Args:
            gemini_api_key: Gemini API 키
        """
        # Gemini 설정
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")

        # MediaPipe 분석기
        self.mediapipe_analyzer = MediaPipeFaceAnalyzer()
        logger.info("✅ MediaPipe 분석기 초기화 완료")

        # 통계
        self.stats = {
            "total_downloaded": 0,
            "mediapipe_success": 0,
            "mediapipe_failed": 0,
            "gemini_success": 0,
            "gemini_failed": 0,
            "total_samples": 0
        }

    def download_ai_face(self) -> Optional[bytes]:
        """
        ThisPersonDoesNotExist.com에서 AI 생성 얼굴 다운로드

        Returns:
            이미지 바이트 (실패 시 None)
        """
        try:
            # 캐시 방지를 위해 타임스탬프 추가
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Cache-Control': 'no-cache'
            }

            response = requests.get(
                self.THISPERSONDOESNOTEXIST_URL,
                headers=headers,
                timeout=10
            )

            if response.status_code == 200:
                self.stats["total_downloaded"] += 1
                logger.info(f"✅ AI 얼굴 다운로드 완료 ({len(response.content)} bytes)")
                return response.content
            else:
                logger.error(f"❌ 다운로드 실패: HTTP {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"❌ 다운로드 오류: {str(e)}")
            return None

    def analyze_with_mediapipe(self, image_data: bytes) -> Optional[MediaPipeFaceFeatures]:
        """
        MediaPipe로 얼굴 분석 (연속형 변수 추출)

        Args:
            image_data: 이미지 바이트

        Returns:
            MediaPipeFaceFeatures 또는 None
        """
        try:
            features = self.mediapipe_analyzer.analyze(image_data)

            if features:
                self.stats["mediapipe_success"] += 1
                logger.info(
                    f"✅ MediaPipe 분석 성공: {features.face_shape} / {features.skin_tone} "
                    f"(신뢰도: {features.confidence:.0%})"
                )
                return features
            else:
                self.stats["mediapipe_failed"] += 1
                logger.warning("⚠️ MediaPipe 얼굴 검출 실패 (재시도 필요)")
                return None

        except Exception as e:
            self.stats["mediapipe_failed"] += 1
            logger.error(f"❌ MediaPipe 분석 오류: {str(e)}")
            return None

    def get_gemini_recommendations(
        self,
        image_data: bytes,
        mp_features: MediaPipeFaceFeatures
    ) -> Optional[Dict]:
        """
        Gemini Vision API로 헤어스타일 추천 받기

        Args:
            image_data: 이미지 바이트
            mp_features: MediaPipe 분석 결과

        Returns:
            {"recommended": [...], "not_recommended": [...]}
        """
        try:
            # PIL Image로 변환
            image = Image.open(io.BytesIO(image_data))

            # 프롬프트 (MediaPipe 결과 힌트 포함)
            prompt = f"""당신은 전문 헤어 스타일리스트입니다. 이 남성의 얼굴을 보고 헤어스타일을 추천해주세요.

**📊 MediaPipe 얼굴 측정 데이터 (수학적 분석):**
- 얼굴형: {mp_features.face_shape}
- 피부톤: {mp_features.skin_tone}
- 얼굴 비율(높이/너비): {mp_features.face_ratio:.2f}
- 이마 너비: {mp_features.forehead_width:.0f}px
- 광대 너비: {mp_features.cheekbone_width:.0f}px
- 턱 너비: {mp_features.jaw_width:.0f}px
- ITA 피부톤: {mp_features.ITA_value:.1f}°
- Hue: {mp_features.hue_value:.1f}

**📝 요청사항:**
1. 이 얼굴에 **가장 잘 어울리는** 남성 헤어스타일 3개 추천
2. 이 얼굴에 **어울리지 않는** 남성 헤어스타일 3개 제시

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
  "reasoning": "간단한 이유 (1-2줄)"
}}

**중요:**
- 한국 남성 미용실에서 실제로 사용하는 스타일명
- MediaPipe 측정값을 참고하되, 시각적 판단도 병행"""

            # Gemini API 호출
            response = self.gemini_model.generate_content(
                [prompt, image],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                )
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

            self.stats["gemini_success"] += 1
            logger.info(
                f"✅ Gemini 추천 성공: {len(result['recommended'])}개 추천, "
                f"{len(result['not_recommended'])}개 비추천"
            )

            return result

        except Exception as e:
            self.stats["gemini_failed"] += 1
            logger.error(f"❌ Gemini 추천 실패: {str(e)}")
            return None

    def create_training_sample(
        self,
        mp_features: MediaPipeFaceFeatures,
        gemini_result: Dict
    ) -> List[Dict]:
        """
        학습 샘플 생성 (라벨링 포함)

        Args:
            mp_features: MediaPipe 분석 결과
            gemini_result: Gemini 추천 결과

        Returns:
            학습 샘플 리스트
        """
        samples = []

        # 연속형 변수 추출
        face_features = [
            mp_features.face_ratio,
            mp_features.forehead_width,
            mp_features.cheekbone_width,
            mp_features.jaw_width,
            mp_features.forehead_width / mp_features.cheekbone_width,  # forehead_ratio
            mp_features.jaw_width / mp_features.cheekbone_width  # jaw_ratio
        ]

        skin_features = [
            mp_features.ITA_value,
            mp_features.hue_value
        ]

        # 1. 추천 스타일 → 순위별 점수 (95/85/75)
        for rank, style in enumerate(gemini_result["recommended"][:3], start=1):
            score = 95 - (rank - 1) * 10

            samples.append({
                "face_shape": mp_features.face_shape,
                "skin_tone": mp_features.skin_tone,
                "face_features": face_features.copy(),
                "skin_features": skin_features.copy(),
                "hairstyle": style,
                "score": score,
                "source": "ai_face_recommended",
                "rank": rank,
                "mediapipe_confidence": mp_features.confidence
            })

        # 2. 비추천 스타일 → 낮은 점수 (10~30)
        for style in gemini_result["not_recommended"][:3]:
            score = np.random.uniform(10, 30)

            samples.append({
                "face_shape": mp_features.face_shape,
                "skin_tone": mp_features.skin_tone,
                "face_features": face_features.copy(),
                "skin_features": skin_features.copy(),
                "hairstyle": style,
                "score": round(score, 1),
                "source": "ai_face_not_recommended",
                "rank": None,
                "mediapipe_confidence": mp_features.confidence
            })

        self.stats["total_samples"] += len(samples)

        return samples

    def collect_batch(self, num_samples: int, delay: float = 2.0) -> List[Dict]:
        """
        배치 데이터 수집

        Args:
            num_samples: 수집할 AI 얼굴 개수
            delay: 각 요청 사이 대기 시간 (초)

        Returns:
            학습 샘플 리스트
        """
        all_samples = []
        successful_faces = 0

        logger.info(f"🚀 데이터 수집 시작: 목표 {num_samples}개 AI 얼굴")

        for i in range(num_samples):
            logger.info(f"\n{'='*60}")
            logger.info(f"[{i+1}/{num_samples}] AI 얼굴 처리 중...")
            logger.info(f"{'='*60}")

            # 1. AI 얼굴 다운로드
            image_data = self.download_ai_face()
            if not image_data:
                logger.warning("재시도...")
                time.sleep(delay)
                continue

            # 2. MediaPipe 분석
            mp_features = self.analyze_with_mediapipe(image_data)
            if not mp_features:
                logger.warning("얼굴 검출 실패, 다음 이미지로...")
                time.sleep(delay)
                continue

            # 3. Gemini 추천
            gemini_result = self.get_gemini_recommendations(image_data, mp_features)
            if not gemini_result:
                logger.warning("Gemini 추천 실패, 다음 이미지로...")
                time.sleep(delay)
                continue

            # 4. 학습 샘플 생성
            samples = self.create_training_sample(mp_features, gemini_result)
            all_samples.extend(samples)

            successful_faces += 1
            logger.info(f"✅ 성공: {len(samples)}개 샘플 생성 (누적: {len(all_samples)}개)")

            # 진행률
            logger.info(
                f"📊 진행률: {successful_faces}/{num_samples} 얼굴 "
                f"({successful_faces/num_samples*100:.1f}%)"
            )

            # API 제한 방지
            time.sleep(delay)

        logger.info(f"\n{'='*60}")
        logger.info(f"🎉 수집 완료!")
        logger.info(f"  - 성공한 얼굴: {successful_faces}/{num_samples}")
        logger.info(f"  - 총 학습 샘플: {len(all_samples)}개")
        logger.info(f"{'='*60}")

        return all_samples

    def save_to_npz(self, samples: List[Dict], output_path: str):
        """
        NPZ 형식으로 저장

        Args:
            samples: 학습 샘플 리스트
            output_path: 출력 경로
        """
        face_features = np.array([s["face_features"] for s in samples], dtype=np.float32)
        skin_features = np.array([s["skin_features"] for s in samples], dtype=np.float32)
        hairstyles = np.array([s["hairstyle"] for s in samples], dtype=object)
        scores = np.array([s["score"] for s in samples], dtype=np.float32)
        metadata = np.array([{
            "face_shape": s["face_shape"],
            "skin_tone": s["skin_tone"],
            "source": s["source"],
            "rank": s["rank"],
            "confidence": s["mediapipe_confidence"]
        } for s in samples], dtype=object)

        np.savez_compressed(
            output_path,
            face_features=face_features,
            skin_features=skin_features,
            hairstyles=hairstyles,
            scores=scores,
            metadata=metadata
        )

        logger.info(f"✅ NPZ 저장 완료: {output_path}")
        logger.info(f"  - Face features: {face_features.shape}")
        logger.info(f"  - Skin features: {skin_features.shape}")
        logger.info(f"  - Hairstyles: {len(hairstyles)}")
        logger.info(f"  - Scores: {scores.shape}")

    def print_statistics(self):
        """통계 출력"""
        logger.info(f"\n{'='*60}")
        logger.info("📊 최종 통계")
        logger.info(f"{'='*60}")
        logger.info(f"  다운로드: {self.stats['total_downloaded']}개")
        logger.info(f"  MediaPipe 성공: {self.stats['mediapipe_success']}개")
        logger.info(f"  MediaPipe 실패: {self.stats['mediapipe_failed']}개")
        logger.info(f"  Gemini 성공: {self.stats['gemini_success']}개")
        logger.info(f"  Gemini 실패: {self.stats['gemini_failed']}개")
        logger.info(f"  총 학습 샘플: {self.stats['total_samples']}개")
        logger.info(f"{'='*60}")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="AI 얼굴로 학습 데이터 수집")
    parser.add_argument(
        "-n", "--num-faces",
        type=int,
        default=100,
        help="수집할 AI 얼굴 개수 (기본: 100)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="각 요청 사이 대기 시간(초) (기본: 2.0)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data_source/ai_face_training_data.npz",
        help="출력 파일 경로"
    )

    args = parser.parse_args()

    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        return

    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 수집기 초기화
    collector = AIFaceDataCollector(api_key)

    # 데이터 수집
    logger.info(f"🎯 목표: {args.num_faces}개 AI 얼굴")
    logger.info(f"🎯 예상 샘플: ~{args.num_faces * 6}개 (얼굴당 6개)")
    logger.info(f"🎯 예상 시간: ~{args.num_faces * args.delay / 60:.1f}분")

    samples = collector.collect_batch(args.num_faces, args.delay)

    if not samples:
        logger.error("❌ 수집된 샘플이 없습니다!")
        return

    # 저장
    collector.save_to_npz(samples, str(output_path))

    # JSON 샘플 저장 (검증용)
    json_path = output_path.parent / f"{output_path.stem}_sample.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(samples[:10], f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"✅ JSON 샘플 저장: {json_path}")

    # 통계 출력
    collector.print_statistics()

    logger.info("\n🎉 완료! 이제 이 데이터로 모델을 학습하세요:")
    logger.info(f"   python scripts/train_model_v4.py --data {output_path}")


if __name__ == "__main__":
    main()
