"""
ONNX 기반 경량 헤어스타일 추천기

PyTorch 없이 ONNX Runtime만으로 추론하여
Lambda 콜드 스타트를 대폭 줄입니다.

의존성:
- onnxruntime (~10MB) vs PyTorch (~180MB)
- numpy

Author: HairMe ML Team
Date: 2025-12-03
Version: 1.0.0
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent

# 라벨 정규화 상수
LABEL_MIN = 10.0
LABEL_MAX = 95.0
LABEL_RANGE = LABEL_MAX - LABEL_MIN

# 학습 데이터 통계 (입력 스케일링용)
FACE_FEATURE_STATS = {
    0: {"min": 0.99, "max": 1.51, "mean": 1.20, "std": 0.06},   # face_ratio
    1: {"min": 301.10, "max": 495.30, "mean": 458.13, "std": 14.31},  # forehead_width
    2: {"min": 421.40, "max": 641.00, "mean": 561.34, "std": 19.73},  # cheekbone_width
    3: {"min": 333.90, "max": 524.10, "mean": 447.70, "std": 19.82},  # jaw_width
    4: {"min": 0.71, "max": 0.89, "mean": 0.82, "std": 0.02},   # forehead_ratio
    5: {"min": 0.73, "max": 0.86, "mean": 0.80, "std": 0.02},   # jaw_ratio
}

SKIN_FEATURE_STATS = {
    0: {"min": 50.53, "max": 89.26, "mean": 79.91, "std": 3.90},  # ITA_value
    1: {"min": 5.96, "max": 142.39, "mean": 12.09, "std": 10.97},  # hue_value
}


def scale_input_features(
    face_features: np.ndarray,
    skin_features: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    추론 입력을 학습 데이터 분포에 맞게 스케일링
    (ml_recommender.py의 scale_input_features와 동일한 로직)
    """
    face_scaled = face_features.copy()
    skin_scaled = skin_features.copy()

    # 1D 배열인 경우
    if len(face_scaled.shape) == 1:
        # 입력 이미지의 스케일 추정 (cheekbone_width 기준)
        input_cheekbone = face_scaled[2]
        train_cheekbone_mean = FACE_FEATURE_STATS[2]["mean"]  # 561.34

        # 스케일 팩터 계산
        if input_cheekbone > 0:
            scale_factor = train_cheekbone_mean / input_cheekbone
        else:
            scale_factor = 1.0

        # 픽셀 기반 특징만 스케일링 (인덱스 1, 2, 3)
        face_scaled[1] = face_features[1] * scale_factor
        face_scaled[2] = face_features[2] * scale_factor
        face_scaled[3] = face_features[3] * scale_factor

        # 스케일링된 값이 학습 데이터 범위 내에 있도록 클리핑
        for idx in [1, 2, 3]:
            min_val = FACE_FEATURE_STATS[idx]["min"]
            max_val = FACE_FEATURE_STATS[idx]["max"]
            face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

        # 비율 특징도 학습 데이터 범위 내에 있도록 클리핑
        for idx in [0, 4, 5]:
            min_val = FACE_FEATURE_STATS[idx]["min"]
            max_val = FACE_FEATURE_STATS[idx]["max"]
            face_scaled[idx] = np.clip(face_scaled[idx], min_val, max_val)

        # 피부 특징 클리핑
        for idx in [0, 1]:
            min_val = SKIN_FEATURE_STATS[idx]["min"]
            max_val = SKIN_FEATURE_STATS[idx]["max"]
            skin_scaled[idx] = np.clip(skin_scaled[idx], min_val, max_val)
    else:
        # 배치 처리
        for i in range(face_scaled.shape[0]):
            face_scaled[i], skin_scaled[i] = scale_input_features(
                face_scaled[i], skin_scaled[i]
            )

    return face_scaled, skin_scaled


def encode_face_shape(face_shape: str) -> np.ndarray:
    """
    얼굴형을 one-hot 인코딩 (6차원 - PyTorch 모델과 동일)
    """
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    vec = np.zeros(6, dtype=np.float32)

    # 하트형은 계란형으로 매핑
    if face_shape == "하트형":
        face_shape = "계란형"

    # 기본 4가지 얼굴형에 대한 one-hot 인코딩
    if face_shape in FACE_SHAPES:
        idx = FACE_SHAPES.index(face_shape)
        vec[idx] = 1.0
    else:
        vec[3] = 1.0  # 계란형 기본값

    # 추가 특징 차원 (모델 학습 시 사용됨)
    vec[4] = 0.5
    vec[5] = 0.5

    return vec


def encode_skin_tone(skin_tone: str) -> np.ndarray:
    """
    피부톤을 one-hot 인코딩 (2차원 - PyTorch 모델과 동일)
    """
    vec = np.zeros(2, dtype=np.float32)

    # 봄/가을 -> 웜톤(0), 여름/겨울 -> 쿨톤(1)
    if skin_tone in ["봄웜", "가을웜"]:
        vec[0] = 1.0  # 웜톤
    elif skin_tone in ["여름쿨", "겨울쿨"]:
        vec[1] = 1.0  # 쿨톤
    else:
        # 알 수 없는 톤이면 중립
        vec[0] = 0.5
        vec[1] = 0.5

    return vec


class ONNXHairstyleRecommender:
    """
    ONNX 기반 경량 헤어스타일 추천기

    PyTorch 없이 동작하여 Lambda 콜드 스타트 최적화
    """

    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(
        self,
        onnx_model_path: str = None,
        embeddings_path: str = None,
        gender_metadata_path: str = None
    ):
        """
        초기화

        Args:
            onnx_model_path: ONNX 모델 경로
            embeddings_path: 헤어스타일 임베딩 경로
            gender_metadata_path: 성별 메타데이터 경로
        """
        # 기본 경로 설정
        if onnx_model_path is None:
            onnx_model_path = PROJECT_ROOT / "models" / "onnx" / "recommendation_model_v6.onnx"
        if embeddings_path is None:
            embeddings_path = PROJECT_ROOT / "data_source" / "style_embeddings.npz"
        if gender_metadata_path is None:
            gender_metadata_path = PROJECT_ROOT / "data_source" / "hairstyle_gender.json"

        self.onnx_path = Path(onnx_model_path)
        self.embeddings_path = Path(embeddings_path)
        self.gender_metadata_path = Path(gender_metadata_path)

        # ONNX 세션 로드
        self._load_onnx_session()

        # 임베딩 로드
        self._load_embeddings()

        # 성별 메타데이터 로드
        self._load_gender_metadata()

        logger.info("✅ ONNX 추천기 초기화 완료")

    def _load_onnx_session(self):
        """ONNX Runtime 세션 로드"""
        try:
            import onnxruntime as ort

            if not self.onnx_path.exists():
                raise FileNotFoundError(f"ONNX 모델 없음: {self.onnx_path}")

            # 세션 옵션 (최적화)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=sess_options,
                providers=['CPUExecutionProvider']
            )

            logger.info(f"✅ ONNX 모델 로드: {self.onnx_path.name}")

        except ImportError:
            logger.error("❌ onnxruntime 패키지 필요: pip install onnxruntime")
            raise
        except Exception as e:
            logger.error(f"❌ ONNX 로드 실패: {e}")
            raise

    def _load_embeddings(self):
        """헤어스타일 임베딩 로드"""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(f"임베딩 파일 없음: {self.embeddings_path}")

        data = np.load(self.embeddings_path, allow_pickle=True)

        # 키 이름 호환성 ('style_names' 또는 'styles')
        if 'style_names' in data:
            self.style_names = list(data['style_names'])
        elif 'styles' in data:
            self.style_names = list(data['styles'])
        else:
            raise KeyError(f"임베딩 파일에 스타일 이름 키가 없습니다. 키: {list(data.keys())}")
        self.style_embeddings = data['embeddings'].astype(np.float32)

        # 인덱스 매핑
        self.style_to_idx = {name: idx for idx, name in enumerate(self.style_names)}
        self.idx_to_style = {idx: name for idx, name in enumerate(self.style_names)}

        logger.info(f"✅ 임베딩 로드: {len(self.style_names)}개 스타일")

    def _load_gender_metadata(self):
        """성별 메타데이터 로드"""
        self.gender_metadata = {}

        if self.gender_metadata_path.exists():
            with open(self.gender_metadata_path, 'r', encoding='utf-8') as f:
                self.gender_metadata = json.load(f)
            logger.info(f"✅ 성별 메타데이터 로드: {len(self.gender_metadata)}개")
        else:
            logger.warning(f"⚠️ 성별 메타데이터 없음: {self.gender_metadata_path}")

    def predict_single(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embedding: np.ndarray
    ) -> float:
        """
        단일 추론

        Args:
            face_features: (6,) or (1, 6)
            skin_features: (2,) or (1, 2)
            style_embedding: (384,) or (1, 384)

        Returns:
            score: 0~1 정규화 점수
        """
        # 배치 차원 추가
        if len(face_features.shape) == 1:
            face_features = face_features.reshape(1, -1)
        if len(skin_features.shape) == 1:
            skin_features = skin_features.reshape(1, -1)
        if len(style_embedding.shape) == 1:
            style_embedding = style_embedding.reshape(1, -1)

        inputs = {
            'face_features': face_features.astype(np.float32),
            'skin_features': skin_features.astype(np.float32),
            'style_embedding': style_embedding.astype(np.float32)
        }

        outputs = self.session.run(None, inputs)
        return float(outputs[0][0])

    def predict_batch(
        self,
        face_features: np.ndarray,
        skin_features: np.ndarray,
        style_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        배치 추론 (개별 추론으로 처리 - ONNX dynamic batch 이슈 회피)

        Args:
            face_features: (batch, 6)
            skin_features: (batch, 2)
            style_embeddings: (batch, 384)

        Returns:
            scores: (batch,) - 0~1 정규화 점수
        """
        batch_size = style_embeddings.shape[0]
        scores = []

        for i in range(batch_size):
            score = self.predict_single(
                face_features[i],
                skin_features[i],
                style_embeddings[i]
            )
            scores.append(score)

        return np.array(scores)

    def recommend_top_k(
        self,
        face_shape: str,
        skin_tone: str,
        k: int = 3,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None
    ) -> List[Dict]:
        """
        Top-K 헤어스타일 추천

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            k: 추천 개수
            face_features: 실제 측정값 [6]
            skin_features: 실제 측정값 [2]
            gender: 성별 필터 ("male", "female", None)

        Returns:
            추천 리스트
        """
        # 특징 벡터 생성 (PyTorch 모델과 동일한 로직)
        if face_features is not None and skin_features is not None:
            # 실제 측정값 사용
            logger.debug(f"[ONNX] 실제 측정값 사용: face={face_features}, skin={skin_features}")
            face_vec = np.array(face_features, dtype=np.float32)
            skin_vec = np.array(skin_features, dtype=np.float32)

            # 스케일링 적용
            face_vec, skin_vec = scale_input_features(face_vec, skin_vec)
            logger.debug(f"[ONNX] 스케일링 후: face={face_vec.tolist()}, skin={skin_vec.tolist()}")
        else:
            # 라벨 기반 인코딩 (PyTorch와 동일한 one-hot 인코딩)
            logger.debug(f"[ONNX] 라벨 기반 인코딩: {face_shape} + {skin_tone}")
            face_vec = encode_face_shape(face_shape)
            skin_vec = encode_skin_tone(skin_tone)
            logger.debug(f"[ONNX] 인코딩 결과: face={face_vec.tolist()}, skin={skin_vec.tolist()}")

        # 배치 추론 준비
        n_styles = len(self.style_names)
        batch_size = 64

        all_scores = []

        for start_idx in range(0, n_styles, batch_size):
            end_idx = min(start_idx + batch_size, n_styles)
            batch_embeddings = self.style_embeddings[start_idx:end_idx]
            curr_batch_size = end_idx - start_idx

            # 입력 복제
            batch_face = np.tile(face_vec, (curr_batch_size, 1))
            batch_skin = np.tile(skin_vec, (curr_batch_size, 1))

            # 추론
            scores = self.predict_batch(batch_face, batch_skin, batch_embeddings)
            all_scores.extend(scores.tolist())

        # 점수를 원래 범위로 역변환 (0~1 → 10~95)
        all_scores = [s * LABEL_RANGE + LABEL_MIN for s in all_scores]

        # 스타일-점수 매핑
        style_scores = [
            {
                "hairstyle_id": idx,
                "hairstyle": self.style_names[idx],
                "score": score
            }
            for idx, score in enumerate(all_scores)
        ]

        # 성별 필터링
        if gender and gender != "neutral":
            filtered = []
            for item in style_scores:
                style_name = item["hairstyle"]
                style_gender = self.gender_metadata.get(style_name, "unisex")

                if gender == "male" and style_gender in ["male", "unisex"]:
                    filtered.append(item)
                elif gender == "female" and style_gender in ["female", "unisex"]:
                    filtered.append(item)

            if filtered:
                style_scores = filtered

        # 점수 내림차순 정렬
        style_scores.sort(key=lambda x: x["score"], reverse=True)

        # 유사도 기반 다양성 필터링
        top_k = []
        for candidate in style_scores:
            if len(top_k) >= k:
                break

            is_similar = False
            for selected in top_k:
                if self._is_similar_style(candidate["hairstyle"], selected["hairstyle"]):
                    is_similar = True
                    break

            if not is_similar:
                top_k.append(candidate)

        return top_k

    def _is_similar_style(self, style_a: str, style_b: str, threshold: float = 0.65) -> bool:
        """두 스타일이 유사한지 확인"""
        ratio = SequenceMatcher(None, style_a, style_b).ratio()
        return ratio >= threshold


# 싱글톤 인스턴스
_onnx_recommender_instance = None


def get_onnx_recommender() -> ONNXHairstyleRecommender:
    """ONNX 추천기 싱글톤"""
    global _onnx_recommender_instance
    if _onnx_recommender_instance is None:
        _onnx_recommender_instance = ONNXHairstyleRecommender()
    return _onnx_recommender_instance


# 테스트
if __name__ == "__main__":
    import sys
    import os

    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ONNX 추천기 테스트")
    print("=" * 60)

    try:
        recommender = ONNXHairstyleRecommender()

        # 추천 테스트
        print("\n추천 테스트:")
        results = recommender.recommend_top_k(
            face_shape="계란형",
            skin_tone="봄웜",
            k=3,
            face_features=[1.2, 450.0, 550.0, 400.0, 0.85, 0.75],
            skin_features=[45.0, 25.0],
            gender="male"
        )

        for i, rec in enumerate(results, 1):
            print(f"  {i}. {rec['hairstyle']}: {rec['score']:.1f}점")

        print("\n✅ ONNX 추천기 테스트 완료!")

    except FileNotFoundError as e:
        print(f"\n⚠️ {e}")
        print("먼저 ONNX 모델을 생성하세요:")
        print("  python scripts/convert_to_onnx.py")

    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
