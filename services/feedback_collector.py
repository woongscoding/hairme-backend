"""
피드백 수집 서비스

사용자 피드백을 수집하고 재학습용 데이터로 저장합니다.
👍 -> 90.0 (user LIKED this combination)
👎 -> 10.0 (user DISLIKED this combination)

Author: HairMe ML Team
Date: 2025-11-13
Version: 1.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 파일 경로
FEEDBACK_JSON_PATH = Path("data/user_feedbacks.json")
FEEDBACK_NPZ_PATH = Path("data/feedback_training_data.npz")
STYLE_EMBEDDINGS_PATH = Path("data_source/style_embeddings.npz")

# 재학습 트리거 임계값
RETRAIN_THRESHOLDS = [500, 1000, 2000, 5000]


class FeedbackCollector:
    """사용자 피드백 수집 및 저장"""

    def __init__(self):
        """초기화"""
        # data 디렉토리 생성
        FEEDBACK_JSON_PATH.parent.mkdir(exist_ok=True)

        # 스타일 임베딩 로드
        self.style_embeddings = None
        self.style_to_idx = {}
        self._load_style_embeddings()

    def _load_style_embeddings(self):
        """헤어스타일 임베딩 로드"""
        try:
            data = np.load(STYLE_EMBEDDINGS_PATH, allow_pickle=False)
            styles = data['styles'].tolist()
            embeddings = data['embeddings']

            self.style_embeddings = embeddings
            self.style_to_idx = {style: idx for idx, style in enumerate(styles)}

            logger.info(f"✅ 스타일 임베딩 로드 완료: {len(styles)}개")

        except Exception as e:
            logger.error(f"❌ 스타일 임베딩 로드 실패: {e}")
            raise

    def get_style_embedding(self, hairstyle_id: int) -> np.ndarray:
        """
        헤어스타일 ID로 임베딩 가져오기

        Args:
            hairstyle_id: 헤어스타일 ID (0-based index)

        Returns:
            임베딩 벡터 (384차원)
        """
        if 0 <= hairstyle_id < len(self.style_embeddings):
            return self.style_embeddings[hairstyle_id]
        else:
            logger.warning(f"⚠️ 잘못된 hairstyle_id: {hairstyle_id}")
            # 기본값으로 0번 임베딩 반환
            return self.style_embeddings[0]

    def save_feedback(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyle_id: int,
        user_reaction: str,
        ml_prediction: float,
        user_id: str = "anonymous"
    ) -> Dict:
        """
        피드백 저장

        Args:
            face_shape: 얼굴형 (예: "계란형")
            skin_tone: 피부톤 (예: "봄웜")
            hairstyle_id: 헤어스타일 ID
            user_reaction: "👍" or "👎"
            ml_prediction: ML 모델 예측 점수
            user_id: 사용자 ID

        Returns:
            {"total_feedbacks": int, "retrain_triggered": bool}
        """
        # 1. Ground truth 결정
        if user_reaction == "👍":
            new_ground_truth = 90.0
            reaction_type = "like"
        elif user_reaction == "👎":
            new_ground_truth = 10.0
            reaction_type = "dislike"
        else:
            raise ValueError(f"Invalid user_reaction: {user_reaction}")

        # 2. 헤어스타일 임베딩 가져오기
        style_embedding = self.get_style_embedding(hairstyle_id)

        # 3. 피드백 데이터 생성
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "face_shape": face_shape,
            "skin_tone": skin_tone,
            "hairstyle_id": int(hairstyle_id),
            "user_reaction": user_reaction,
            "reaction_type": reaction_type,
            "ml_prediction": float(ml_prediction),
            "ground_truth": float(new_ground_truth)
        }

        # 4. JSON 파일에 추가 저장
        self._append_to_json(feedback_entry)

        # 5. NumPy 파일에 추가 저장
        self._append_to_npz(
            face_shape=face_shape,
            skin_tone=skin_tone,
            style_embedding=style_embedding,
            ground_truth=new_ground_truth
        )

        # 6. 전체 피드백 수 확인
        total_count = self._get_total_count()

        # 7. 재학습 트리거 확인
        retrain_triggered = total_count in RETRAIN_THRESHOLDS

        logger.info(
            f"✅ 피드백 저장 완료: {face_shape} + {skin_tone} + ID#{hairstyle_id} "
            f"-> {user_reaction} (GT: {new_ground_truth}) | "
            f"Total: {total_count}, Retrain: {retrain_triggered}"
        )

        return {
            "total_feedbacks": total_count,
            "retrain_triggered": retrain_triggered
        }

    def _append_to_json(self, feedback_entry: Dict):
        """JSON 파일에 피드백 추가"""
        try:
            # 기존 데이터 로드
            if FEEDBACK_JSON_PATH.exists():
                with open(FEEDBACK_JSON_PATH, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
            else:
                feedbacks = []

            # 새 피드백 추가
            feedbacks.append(feedback_entry)

            # 저장
            with open(FEEDBACK_JSON_PATH, 'w', encoding='utf-8') as f:
                json.dump(feedbacks, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"❌ JSON 저장 실패: {e}")
            raise

    def _append_to_npz(
        self,
        face_shape: str,
        skin_tone: str,
        style_embedding: np.ndarray,
        ground_truth: float
    ):
        """NumPy 파일에 피드백 추가 (학습용)"""
        try:
            # Face shape encoding (one-hot)
            FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
            face_vec = np.zeros(4, dtype=np.float32)

            if face_shape == "하트형":
                face_shape = "계란형"

            if face_shape in FACE_SHAPES:
                idx = FACE_SHAPES.index(face_shape)
                face_vec[idx] = 1.0
            else:
                face_vec[3] = 1.0  # 기본값: 계란형

            # Skin tone encoding (one-hot)
            SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]
            tone_vec = np.zeros(4, dtype=np.float32)

            if skin_tone in SKIN_TONES:
                idx = SKIN_TONES.index(skin_tone)
                tone_vec[idx] = 1.0
            else:
                tone_vec[2] = 1.0  # 기본값: 봄웜

            # Feature vector: [face(4) + tone(4) + style(384)] = 392
            feature = np.concatenate([face_vec, tone_vec, style_embedding])

            # 기존 데이터 로드
            if FEEDBACK_NPZ_PATH.exists():
                data = np.load(FEEDBACK_NPZ_PATH)
                X = data['X']
                y = data['y']

                # 새 데이터 추가
                X = np.vstack([X, feature.reshape(1, -1)])
                y = np.append(y, ground_truth)
            else:
                # 새로 생성
                X = feature.reshape(1, -1)
                y = np.array([ground_truth], dtype=np.float32)

            # 저장
            np.savez_compressed(
                FEEDBACK_NPZ_PATH,
                X=X.astype(np.float32),
                y=y.astype(np.float32)
            )

        except Exception as e:
            logger.error(f"❌ NPZ 저장 실패: {e}")
            raise

    def _get_total_count(self) -> int:
        """전체 피드백 수 반환"""
        try:
            if FEEDBACK_JSON_PATH.exists():
                with open(FEEDBACK_JSON_PATH, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
                return len(feedbacks)
            else:
                return 0
        except Exception as e:
            logger.error(f"❌ 피드백 수 조회 실패: {e}")
            return 0


# ========== 싱글톤 인스턴스 ==========
_collector_instance = None


def get_feedback_collector() -> FeedbackCollector:
    """
    피드백 수집기 싱글톤 인스턴스

    Returns:
        FeedbackCollector 인스턴스
    """
    global _collector_instance

    if _collector_instance is None:
        logger.info("🔧 피드백 수집기 초기화 중...")
        _collector_instance = FeedbackCollector()
        logger.info("✅ 피드백 수집기 준비 완료")

    return _collector_instance
