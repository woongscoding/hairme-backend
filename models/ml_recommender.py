"""
ML 기반 독립형 헤어스타일 추천기

MediaPipe 분석 결과 (얼굴형 + 피부톤)로 학습된 ML 모델을 사용해
모든 헤어스타일의 추천 점수를 예측하고 Top-K를 반환

Author: HairMe ML Team
Date: 2025-11-08
Version: 1.1.0 (Real-time Embedding Support)
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Tuple, TYPE_CHECKING
import logging
import sys
from difflib import SequenceMatcher

# TYPE_CHECKING을 사용하여 런타임에는 import하지 않음
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)


class AttentionLayer(nn.Module):
    """Multi-head self-attention layer"""

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        # Residual connection + layer norm
        x = self.norm(x + self.dropout(attn_out))
        return x


class RecommendationModel(nn.Module):
    """
    연속형 변수 기반 추천 모델 v4

    입력:
    - face_features: [batch, 6] - MediaPipe 얼굴 측정값
    - skin_features: [batch, 2] - MediaPipe 피부 측정값
    - style_emb: [batch, 384] - 헤어스타일 임베딩
    """

    def __init__(
        self,
        face_feat_dim: int = 6,
        skin_feat_dim: int = 2,
        style_embed_dim: int = 384,
        use_attention: bool = True,
        dropout_rate: float = 0.3
    ):
        super().__init__()

        self.face_feat_dim = face_feat_dim
        self.skin_feat_dim = skin_feat_dim
        self.style_embed_dim = style_embed_dim

        # Input projection layers
        self.face_projection = nn.Sequential(
            nn.Linear(face_feat_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        self.skin_projection = nn.Sequential(
            nn.Linear(skin_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5)
        )

        # Total dimension after projection
        self.total_dim = 64 + 32 + style_embed_dim  # 96 + 384 = 480

        # Attention layer
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionLayer(
                embed_dim=self.total_dim,
                num_heads=8,
                dropout=0.1
            )

        # Feature fusion network
        self.fc1 = nn.Linear(self.total_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)

        # Residual connection
        self.residual_proj = nn.Linear(self.total_dim, 128)

        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(dropout_rate * 0.5)

        self.fc4 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def forward(
        self,
        face_features: torch.Tensor,
        skin_features: torch.Tensor,
        style_emb: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass"""
        # Project features
        face_proj = self.face_projection(face_features)
        skin_proj = self.skin_projection(skin_features)

        # Concatenate all features
        x = torch.cat([face_proj, skin_proj, style_emb], dim=1)

        # Apply attention if enabled
        if self.use_attention:
            x_att = x.unsqueeze(1)
            x_att = self.attention(x_att)
            x = x_att.squeeze(1)

        # Store for residual
        residual = self.residual_proj(x)

        # Main network
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # Add residual connection
        x = x + residual

        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc_out(x)

        # 스케일링 적용 (학습 시 30~90점 범위)
        # 클램핑 제거 - 원본 점수를 유지하여 Top-K 내에서 Min-Max 정규화 가능하게 함
        x = (x - 29.0) * 7.5 + 60.0
        # 참고: 클램핑은 recommend_top_k에서 Min-Max 정규화 후 적용

        return x.squeeze(-1)


class MLHairstyleRecommender:
    """ML 기반 헤어스타일 추천기"""

    # MediaPipe와 호환되는 카테고리
    FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
    SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]

    def __init__(
        self,
        model_path: str = "models/hairstyle_recommender_v4_no_leakage.pt",
        embeddings_path: str = "data_source/style_embeddings.npz",
        gender_metadata_path: str = "data_source/hairstyle_gender.json"
    ):
        """
        초기화

        Args:
            model_path: 학습된 모델 경로
            embeddings_path: 헤어스타일 임베딩 경로
            gender_metadata_path: 헤어스타일 성별 메타데이터 경로
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 1. 모델 로드
        logger.info(f"📂 ML 모델 로딩: {model_path}")
        self.model = RecommendationModel()

        # 체크포인트 형식으로 저장된 경우 처리
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 체크포인트 형식
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"✅ 체크포인트에서 모델 로드 완료 (epoch: {checkpoint.get('epoch', 'N/A')})")
            else:
                # 일반 state_dict 형식
                self.model.load_state_dict(checkpoint)
                logger.info(f"✅ 모델 로드 완료")
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {str(e)}")
            raise

        self.model.to(self.device)
        self.model.eval()  # 추론 모드
        logger.info(f"✅ 모델 준비 완료 (디바이스: {self.device})")

        # 2. 헤어스타일 임베딩 로드
        logger.info(f"📂 임베딩 로딩: {embeddings_path}")
        try:
            data = np.load(embeddings_path, allow_pickle=False)
            self.styles = data['styles'].tolist()  # 헤어스타일명 리스트
            self.embeddings = data['embeddings']  # (N, 384) 임베딩
            logger.info(f"✅ 임베딩 로드 완료: {len(self.styles)}개 스타일")
        except Exception as e:
            logger.error(f"❌ 임베딩 로드 실패: {str(e)}")
            raise

        # 스타일명 -> 인덱스 매핑
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

        # 3. 성별 메타데이터 로드 (NEW)
        logger.info(f"📂 성별 메타데이터 로딩: {gender_metadata_path}")
        try:
            import json
            import os
            if os.path.exists(gender_metadata_path):
                with open(gender_metadata_path, 'r', encoding='utf-8') as f:
                    self.gender_metadata = json.load(f)
                logger.info(f"✅ 성별 메타데이터 로드 완료: {len(self.gender_metadata)}개 스타일")
            else:
                logger.warning(f"⚠️ 성별 메타데이터 파일 없음 - 성별 필터링 비활성화")
                self.gender_metadata = {}
        except Exception as e:
            logger.error(f"❌ 성별 메타데이터 로드 실패: {str(e)}")
            self.gender_metadata = {}

        # 4. 실시간 임베딩용 SentenceTransformer 로드 (Lambda에서는 스킵)
        import os
        is_lambda = os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

        if not is_lambda:
            logger.info("🔄 실시간 임베딩 모델 로딩 (paraphrase-multilingual-MiniLM-L12-v2)...")
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("✅ 실시간 임베딩 모델 준비 완료")
            except Exception as e:
                logger.error(f"❌ 실시간 임베딩 모델 로드 실패: {str(e)}")
                self.sentence_model = None
        else:
            logger.info("🔧 Lambda 환경 - 실시간 임베딩 모델 스킵")
            self.sentence_model = None

    def _encode_face_shape(self, face_shape: str) -> np.ndarray:
        """얼굴형을 one-hot 인코딩 (6차원 - 모델과 일치)"""
        vec = np.zeros(6, dtype=np.float32)

        # 하트형은 계란형으로 매핑
        if face_shape == "하트형":
            face_shape = "계란형"
            logger.debug("하트형을 계란형으로 매핑")

        # 기본 4가지 얼굴형에 대한 one-hot 인코딩
        if face_shape in self.FACE_SHAPES:
            idx = self.FACE_SHAPES.index(face_shape)
            vec[idx] = 1.0
        else:
            logger.warning(f"알 수 없는 얼굴형: {face_shape}, 계란형으로 기본값 사용")
            vec[3] = 1.0  # 계란형

        # 추가 특징 차원 (모델 학습 시 사용됨)
        vec[4] = 0.5  # 중간 값으로 초기화
        vec[5] = 0.5  # 중간 값으로 초기화

        return vec

    def _encode_skin_tone(self, skin_tone: str) -> np.ndarray:
        """피부톤을 one-hot 인코딩 (2차원 - 모델과 일치)"""
        vec = np.zeros(2, dtype=np.float32)

        # 봄/가을 -> 웜톤(0), 여름/겨울 -> 쿨톤(1)
        if skin_tone in ["봄웜", "가을웜"]:
            vec[0] = 1.0  # 웜톤
        elif skin_tone in ["여름쿨", "겨울쿨"]:
            vec[1] = 1.0  # 쿨톤
        else:
            logger.warning(f"알 수 없는 피부톤: {skin_tone}, 웜톤으로 기본값 사용")
            vec[0] = 1.0  # 웜톤

        return vec

    def _is_similar_style(self, style_a: str, style_b: str, threshold: float = 0.65) -> bool:
        """
        두 스타일명의 유사도 계산 (0~1)

        Args:
            style_a: 첫 번째 스타일명
            style_b: 두 번째 스타일명
            threshold: 유사도 임계값 (기본 0.65 = 65%)

        Returns:
            threshold 이상이면 True (유사한 스타일)

        Examples:
            - "가르마 스타일 (5:5 또는 6:4)" vs "가르마 스타일 (6:4 또는 7:3)" → 0.74 → True (유사함)
            - "가르마 스타일" vs "가일 컷" → 0.25 → False (다름)
        """
        ratio = SequenceMatcher(None, style_a, style_b).ratio()
        return ratio >= threshold

    def _get_style_embedding(self, style_name: str) -> np.ndarray:
        """
        스타일 임베딩 가져오기 (DB 조회 또는 실시간 생성)

        Args:
            style_name: 헤어스타일명 (정규화된 이름 권장)

        Returns:
            임베딩 벡터 (384,) 또는 None
        """
        # 1. DB 조회 (Fast Path)
        if style_name in self.style_to_idx:
            idx = self.style_to_idx[style_name]
            return self.embeddings[idx]

        # 2. 실시간 생성 (Slow Path)
        if self.sentence_model:
            logger.info(f"🆕 새로운 스타일 발견: '{style_name}' -> 실시간 임베딩 생성")
            try:
                embedding = self.sentence_model.encode(style_name)
                return embedding
            except Exception as e:
                logger.error(f"❌ 임베딩 생성 실패 ({style_name}): {str(e)}")
                return None

        return None

    def predict_score(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyle: str
    ) -> float:
        """
        특정 헤어스타일의 추천 점수 예측 (띄어쓰기 정규화 적용)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyle: 헤어스타일명

        Returns:
            추천 점수 (0-100)
        """
        # 띄어쓰기 정규화 적용
        normalized_style = normalize_style_name(hairstyle)

        # 임베딩 가져오기 (DB or 실시간)
        style_embedding = self._get_style_embedding(normalized_style)

        if style_embedding is None:
            # 원본 이름으로도 시도
            style_embedding = self._get_style_embedding(hairstyle)
            
            if style_embedding is None:
                logger.warning(f"임베딩 생성 불가: '{hairstyle}'")
                return 0.0

        # 개별 특징 벡터 생성
        face_vec = self._encode_face_shape(face_shape)  # (4,)
        tone_vec = self._encode_skin_tone(skin_tone)    # (4,)

        # 모델 추론 - 3개의 개별 텐서로 전달
        with torch.no_grad():
            face_tensor = torch.FloatTensor(face_vec).unsqueeze(0).to(self.device)
            skin_tensor = torch.FloatTensor(tone_vec).unsqueeze(0).to(self.device)
            style_tensor = torch.FloatTensor(style_embedding).unsqueeze(0).to(self.device)

            score_tensor = self.model(face_tensor, skin_tensor, style_tensor)
            score = score_tensor.cpu().item()

        # 0-100 범위로 클리핑
        score = max(0.0, min(100.0, score))

        return round(score, 2)

    def recommend_top_k(
        self,
        face_shape: str = None,
        skin_tone: str = None,
        k: int = 3,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None
    ) -> List[Dict[str, any]]:
        """
        Top-K 헤어스타일 추천 (성별 필터링 적용)

        Args:
            face_shape: 얼굴형 (예: "계란형") - DEPRECATED, 하위 호환성을 위해 유지
            skin_tone: 피부톤 (예: "봄웜") - DEPRECATED, 하위 호환성을 위해 유지
            k: 추천 개수
            face_features: MediaPipe 얼굴 측정값 [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
            skin_features: MediaPipe 피부 측정값 [ITA_value, hue_value] (2차원)
            gender: 성별 ("male", "female", "neutral") - MediaPipe로 추론된 값

        Returns:
            추천 리스트 [{"hairstyle": "...", "score": 85.3}, ...]
        """
        # 실제 측정값 우선 사용, 없으면 라벨 기반 인코딩 (하위 호환성)
        if face_features is not None and skin_features is not None:
            logger.info(f"[ML DEBUG] ML 추천 시작 (실제 측정값 사용) - Top-{k}")
            logger.info(f"[ML DEBUG] Face features: {face_features}")
            logger.info(f"[ML DEBUG] Skin features: {skin_features}")

            # NumPy 배열로 변환
            face_vec = np.array(face_features, dtype=np.float32)
            tone_vec = np.array(skin_features, dtype=np.float32)

            # 차원 검증
            if face_vec.shape[0] != 6:
                raise ValueError(f"face_features는 6차원이어야 합니다. 현재: {face_vec.shape[0]}")
            if tone_vec.shape[0] != 2:
                raise ValueError(f"skin_features는 2차원이어야 합니다. 현재: {tone_vec.shape[0]}")
        else:
            # 하위 호환성: 라벨 기반 인코딩
            logger.warning(f"[ML DEPRECATED] 라벨 기반 인코딩 사용: {face_shape} + {skin_tone}")
            logger.warning("[ML DEPRECATED] 실제 측정값(face_features, skin_features)을 전달하는 것을 권장합니다.")

            if face_shape is None or skin_tone is None:
                raise ValueError("face_features와 skin_features가 없으면 face_shape과 skin_tone을 제공해야 합니다.")

            face_vec = self._encode_face_shape(face_shape)  # (6,)
            tone_vec = self._encode_skin_tone(skin_tone)    # (2,)

        logger.info(f"[ML DEBUG] Face vector: {face_vec.tolist()}")
        logger.info(f"[ML DEBUG] Skin vector: {tone_vec.tolist()}")

        # 모든 헤어스타일에 대해 점수 예측
        all_scores = []

        # 배치 처리로 최적화
        batch_size = 64
        num_styles = len(self.styles)

        for i in range(0, num_styles, batch_size):
            batch_end = min(i + batch_size, num_styles)
            batch_size_actual = batch_end - i
            batch_embeddings = self.embeddings[i:batch_end]

            # 배치 추론 - 3개의 개별 텐서로 전달
            with torch.no_grad():
                # 얼굴형과 피부톤은 배치 크기만큼 복제
                face_batch = np.tile(face_vec, (batch_size_actual, 1))
                skin_batch = np.tile(tone_vec, (batch_size_actual, 1))

                face_tensor = torch.FloatTensor(face_batch).to(self.device)
                skin_tensor = torch.FloatTensor(skin_batch).to(self.device)
                style_tensor = torch.FloatTensor(batch_embeddings).to(self.device)

                # 첫 번째 배치에서만 디버그 정보 출력
                if i == 0:
                    logger.info(f"[ML DEBUG] First batch embedding shape: {batch_embeddings.shape}")
                    logger.info(f"[ML DEBUG] First style embedding std: {batch_embeddings.std():.6f}")
                    logger.info(f"[ML DEBUG] First 3 styles: {self.styles[i:i+3]}")

                scores_tensor = self.model(face_tensor, skin_tensor, style_tensor)
                scores = scores_tensor.cpu().numpy().flatten()

                # 첫 번째 배치에서만 점수 디버그
                if i == 0:
                    logger.info(f"[ML DEBUG] First batch scores: {scores[:5].tolist()}")
                    logger.info(f"[ML DEBUG] Scores std: {scores.std():.6f}")

            # 결과 저장 (원본 점수 그대로 저장)
            for j, score in enumerate(scores):
                style_idx = i + j
                all_scores.append({
                    "hairstyle_id": style_idx,  # ✅ DB ID 추가
                    "hairstyle": self.styles[style_idx],
                    "score": float(score),  # 원본 점수 그대로 저장
                    "original_score": float(score)  # 피드백용 원본 점수 보존
                })

        # 점수 기준 정렬
        all_scores.sort(key=lambda x: x['score'], reverse=True)

        # 성별 필터링 (NEW)
        if gender and self.gender_metadata:
            logger.info(f"[GENDER] 성별 필터링 시작 (gender={gender})")
            filtered_scores = []
            for item in all_scores:
                style_name = item['hairstyle']
                style_gender = self.gender_metadata.get(style_name, "unisex")

                # 성별 매칭 로직:
                # - neutral (애매한 경우): 모든 스타일 추천
                # - male: male + unisex 추천
                # - female: female + unisex 추천
                if gender == "neutral":
                    filtered_scores.append(item)
                elif gender == "male" and style_gender in ["male", "unisex"]:
                    filtered_scores.append(item)
                elif gender == "female" and style_gender in ["female", "unisex"]:
                    filtered_scores.append(item)

            logger.info(
                f"[GENDER] 필터링 완료: {len(all_scores)}개 → {len(filtered_scores)}개 "
                f"(제외: {len(all_scores) - len(filtered_scores)}개)"
            )
            all_scores = filtered_scores
        else:
            logger.info("[GENDER] 성별 필터링 비활성화 (gender 미제공 또는 메타데이터 없음)")

        # 유사도 기반 다양성 필터링 (65% 이상 유사한 스타일 제외)
        top_k_recommendations = []
        similarity_threshold = 0.65
        max_candidates = min(100, len(all_scores))  # 상위 100개까지 탐색

        logger.info(f"[DIVERSITY] 다양성 필터링 시작 (threshold={similarity_threshold})")

        for candidate in all_scores[:max_candidates]:
            if len(top_k_recommendations) >= k:
                break

            candidate_style = candidate['hairstyle']

            # 이미 선택된 스타일과 유사도 체크
            is_duplicate = False
            for selected in top_k_recommendations:
                selected_style = selected['hairstyle']
                if self._is_similar_style(candidate_style, selected_style, similarity_threshold):
                    logger.debug(
                        f"[DIVERSITY] 유사한 스타일 제외: '{candidate_style}' "
                        f"(유사: '{selected_style}')"
                    )
                    is_duplicate = True
                    break

            if not is_duplicate:
                top_k_recommendations.append(candidate)
                logger.info(
                    f"[DIVERSITY] 선택 ({len(top_k_recommendations)}/{k}): "
                    f"'{candidate_style}' (점수: {candidate['score']:.2f})"
                )

        # k개를 채우지 못한 경우 경고
        if len(top_k_recommendations) < k:
            logger.warning(
                f"[DIVERSITY] 다양한 스타일 {k}개를 찾지 못함 "
                f"(실제: {len(top_k_recommendations)}개). "
                f"threshold를 낮추거나 데이터를 확인하세요."
            )

        # Min-Max 정규화를 사용한 점수 스케일링
        # Top-K 내에서 점수를 75~95점 범위로 정규화하여 차별화된 점수 제공
        if len(top_k_recommendations) >= 2:
            raw_scores = [rec['original_score'] for rec in top_k_recommendations]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)

            # 점수 차이가 있는 경우에만 정규화
            if max_raw > min_raw:
                # 목표 범위: 75 ~ 95점
                target_min, target_max = 75.0, 95.0

                logger.info(f"[SCORE NORM] Raw scores: {raw_scores}")
                logger.info(f"[SCORE NORM] Raw range: {min_raw:.2f} ~ {max_raw:.2f}")

                for rec in top_k_recommendations:
                    raw = rec['original_score']
                    # Min-Max 정규화: (raw - min) / (max - min) * (target_max - target_min) + target_min
                    normalized = (raw - min_raw) / (max_raw - min_raw) * (target_max - target_min) + target_min
                    rec['score'] = round(normalized, 2)

                logger.info(f"[SCORE NORM] Normalized scores: {[r['score'] for r in top_k_recommendations]}")
            else:
                # 모든 점수가 동일한 경우 (드물지만) 중간값 사용
                for i, rec in enumerate(top_k_recommendations):
                    rec['score'] = round(95.0 - i * 3, 2)  # 95, 92, 89...
                logger.info(f"[SCORE NORM] Same scores - using fallback: {[r['score'] for r in top_k_recommendations]}")
        elif len(top_k_recommendations) == 1:
            # 1개만 있는 경우
            top_k_recommendations[0]['score'] = 90.0
            logger.info("[SCORE NORM] Single recommendation - set to 90.0")

        # 디버그: Top-K 점수 분포
        if top_k_recommendations:
            scores_list = [r['score'] for r in top_k_recommendations]
            logger.info(f"[ML DEBUG] Top-{k} final scores: {scores_list}")
            logger.info(f"[ML DEBUG] Score range: {min(scores_list):.2f} ~ {max(scores_list):.2f}")

        logger.info(
            f"[ML RESULT] ML 추천 완료: {[r['hairstyle'] for r in top_k_recommendations]}"
        )

        return top_k_recommendations

    def batch_predict(
        self,
        face_shape: str,
        skin_tone: str,
        hairstyles: List[str]
    ) -> Dict[str, float]:
        """
        여러 헤어스타일의 점수를 한 번에 예측 (띄어쓰기 정규화 적용)

        Args:
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyles: 헤어스타일 리스트

        Returns:
            {헤어스타일: 점수} 딕셔너리
        """
        results = {}
        
        # 1. 임베딩 수집 (DB or 실시간)
        valid_styles = []
        batch_embeddings = []
        
        for style in hairstyles:
            normalized = normalize_style_name(style)
            embedding = self._get_style_embedding(normalized)
            
            if embedding is None:
                # 원본 이름으로도 시도
                embedding = self._get_style_embedding(style)
            
            if embedding is not None:
                valid_styles.append(style)
                batch_embeddings.append(embedding)
            else:
                logger.warning(f"임베딩 생성 불가로 건너뜀: {style}")

        if not valid_styles:
            logger.warning("유효한 헤어스타일이 없습니다")
            return results

        # 2. 얼굴형과 피부톤 특징 벡터 생성
        face_vec = self._encode_face_shape(face_shape)  # (4,)
        tone_vec = self._encode_skin_tone(skin_tone)    # (4,)

        batch_embeddings = np.array(batch_embeddings, dtype=np.float32)

        # 3. 배치 추론 - 3개의 개별 텐서로 전달
        with torch.no_grad():
            batch_size = len(valid_styles)
            face_batch = np.tile(face_vec, (batch_size, 1))
            skin_batch = np.tile(tone_vec, (batch_size, 1))

            face_tensor = torch.FloatTensor(face_batch).to(self.device)
            skin_tensor = torch.FloatTensor(skin_batch).to(self.device)
            style_tensor = torch.FloatTensor(batch_embeddings).to(self.device)

            scores_tensor = self.model(face_tensor, skin_tensor, style_tensor)
            scores = scores_tensor.cpu().numpy().flatten()

        # 4. 결과 저장
        for style, score in zip(valid_styles, scores):
            results[style] = round(max(0.0, min(100.0, float(score))), 2)

        return results


# ========== 싱글톤 인스턴스 (전역 사용) ==========
_recommender_instance = None


def get_ml_recommender() -> MLHairstyleRecommender:
    """
    ML 추천기 싱글톤 인스턴스 가져오기

    Returns:
        MLHairstyleRecommender 인스턴스
    """
    global _recommender_instance

    if _recommender_instance is None:
        logger.info("🔧 ML 추천기 초기화 중...")
        _recommender_instance = MLHairstyleRecommender()
        logger.info("✅ ML 추천기 준비 완료")

    return _recommender_instance
