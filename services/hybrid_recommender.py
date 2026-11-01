"""
ML 기반 헤어스타일 추천 서비스

MediaPipe 얼굴 분석 + ML 모델을 사용한 추천 제공

Author: HairMe ML Team
Date: 2025-11-08
Version: 2.0.0 (ML-only mode)
"""

import logging
import statistics
from typing import List, Dict, Optional, Any, Tuple
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from services import rec_exploration
from utils.style_preprocessor import normalize_style_name

logger = logging.getLogger(__name__)


def _raw_score(rec: Dict[str, Any]) -> float:
    """정규화·반올림 전 원점수 (10~95). 없으면 score"""
    raw = rec.get("original_score")
    return float(raw if raw is not None else rec.get("score", 0.0))


def _align_full_ranking(
    full_ranking: List[Dict[str, Any]], recommendations: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """full_ranking 맨 앞을 화면에 나온 ML 카드(표시 순서)로 맞춘다

    나머지는 기존 점수순 그대로 두고 앞에 올린 스타일은 중복 제거한다.
    전체 랭킹 계산이 꺼졌거나 실패해 비어 있으면 그대로 둔다.
    """
    if not full_ranking:
        return full_ranking

    by_name = {}
    for entry in full_ranking:
        by_name.setdefault(normalize_style_name(entry["style_name"]), entry)

    head: List[Dict[str, Any]] = []
    shown = set()
    for card in recommendations:
        if card.get("source") != "ml":
            continue
        name = (card.get("style_name") or "").strip()
        normalized = normalize_style_name(name) if name else ""
        if not normalized or normalized in shown:
            continue
        shown.add(normalized)
        head.append(
            by_name.get(normalized) or {"style_name": name, "score": card.get("score")}
        )

    tail = [
        entry
        for entry in full_ranking
        if normalize_style_name(entry["style_name"]) not in shown
    ]
    return head + tail


class MLRecommendationService:
    """ML 기반 헤어스타일 추천 서비스"""

    def __init__(self):
        """초기화"""
        # ML 추천기 로드 (Lazy import)
        try:
            from models.ml_recommender import get_ml_recommender

            self.ml_recommender = get_ml_recommender()
            self.ml_available = True
            logger.info("✅ ML 추천기 로드 성공")
        except Exception as e:
            logger.error(f"❌ ML 추천기 로드 실패: {str(e)}")
            self.ml_recommender = None
            self.ml_available = False
            raise

        # 추천 이유 생성기 로드 (Lazy import)
        try:
            from services.reason_generator import get_reason_generator

            self.reason_generator = get_reason_generator()
            logger.info("✅ 추천 이유 생성기 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️ 추천 이유 생성기 로드 실패: {str(e)}")
            self.reason_generator = None

    @property
    def model_version(self) -> str:
        """현재 로드된 추천 모델의 버전 (체크포인트 config["version"])

        재학습 모델은 "v6_feedback_YYYYMMDD" 형태다. 계측 로그/DynamoDB 저장에
        쓰이며, 추천기가 없거나 속성이 없으면 "unknown" 을 돌려준다.
        """
        return str(getattr(self.ml_recommender, "model_version", None) or "unknown")

    @staticmethod
    def _score_stats(ml_recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """ML 원점수(0-100)의 1위 점수와 표준편차

        응답의 score 는 0-1 로 나눈 뒤 소수 2자리로 반올림하므로 (1점 단위),
        표준편차가 2~5 수준인 현재 모델에서는 해상도가 부족하다.
        계측은 반올림 전 원점수로 계산한다.
        """
        scores = [
            float(rec.get("score"))
            for rec in ml_recommendations
            if rec.get("score") is not None
        ]
        if not scores:
            return {"top_score": None, "score_stddev": None}
        return {
            "top_score": round(max(scores), 2),
            "score_stddev": round(statistics.pstdev(scores), 3),
        }

    def _build_recommendations(
        self,
        ml_recommendations: List[Dict[str, Any]],
        face_shape: str,
        skin_tone: str,
        exposure: Optional[List[Tuple[int, float]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        ML 추천 결과를 응답 형식으로 변환

        Args:
            ml_recommendations: ML 추천 리스트
            face_shape: 얼굴형
            skin_tone: 피부톤
            exposure: ml_recommendations 와 같은 순서의 (ml_rank, propensity).
                None 이면 결정적 노출 (ml_rank = 순서, propensity = 1.0)

        Returns:
            추천 리스트
        """
        result = []
        seen_styles = set()

        for position, rec in enumerate(ml_recommendations):
            hairstyle_id = rec.get("hairstyle_id")
            style_name = rec.get("hairstyle", "").strip()
            ml_score = rec.get("score", 0.0)

            if not style_name:
                continue

            # 띄어쓰기 정규화 적용 (중복 검사용)
            normalized_name = normalize_style_name(style_name)

            if normalized_name in seen_styles:
                continue

            ml_rank, propensity = (
                exposure[position] if exposure is not None else (position + 1, 1.0)
            )

            # 템플릿 기반 이유 생성
            if self.reason_generator:
                try:
                    reason = self.reason_generator.generate_with_score(
                        face_shape, skin_tone, style_name, ml_score
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 이유 생성 실패: {str(e)}")
                    reason = f"ML 모델 추천 (점수: {ml_score:.1f})"
            else:
                reason = f"ML 모델 추천 (점수: {ml_score:.1f})"

            result.append(
                {
                    "hairstyle_id": hairstyle_id,
                    "style_name": style_name,
                    "reason": reason,
                    "source": "ml",
                    "score": round(
                        ml_score / 100.0, 2
                    ),  # 0-1 범위로 변환 (안드로이드 호환)
                    "rank": len(result) + 1,
                    # 노출 메타 (재학습 순위 편향 보정용). score 는 1점 해상도라
                    # 원점수를 따로 남기고, 표시 순위 rank 와 풀 안 순위를 구분한다.
                    "raw_score": round(_raw_score(rec), 2),
                    "ml_rank": ml_rank,
                    "propensity": propensity,
                }
            )

            seen_styles.add(normalized_name)

        logger.info(f"✅ ML 추천 결과: {len(result)}개")

        return result

    def _full_ranking(
        self,
        face_shape: str,
        skin_tone: str,
        face_features: Optional[List[float]],
        skin_features: Optional[List[float]],
        gender: Optional[str],
    ) -> List[Dict[str, Any]]:
        """전체 스타일 랭킹 [{style_name, score}] 점수순 (보상형 광고 해제 기능)

        Top-3 호출과 별도로 k=FULL_RANKING_LIMIT 로 한 번 더 점수를 매긴다.
        같은 호출로 합치면 Top-3 결과가 달라질 수 있다 - 후보 폴백(allowed < k)과
        v4 모델의 Top-K 내 정규화가 k 에 의존하기 때문.
        score 는 recommendations 와 같은 0-1 스케일이다. 실패하면 빈 배열.
        """
        limit = settings.FULL_RANKING_LIMIT
        if limit <= 0 or not (self.ml_available and self.ml_recommender):
            return []

        # 노이즈 제외 후보보다 k 가 크면 recommend_top_k 가 전체 후보로 폴백하므로 맞춘다
        allowed = getattr(self.ml_recommender, "allowed_indices", None)
        if isinstance(allowed, list) and allowed:
            limit = min(limit, len(allowed))

        try:
            ranked = self.ml_recommender.recommend_top_k(
                face_shape=face_shape,
                skin_tone=skin_tone,
                k=limit,
                face_features=face_features,
                skin_features=skin_features,
                gender=gender,
                log_selection=False,
            )
        except Exception as e:
            logger.warning(f"⚠️ 전체 랭킹 계산 실패: {str(e)}")
            return []

        ranking: List[Dict[str, Any]] = []
        seen = set()
        for rec in ranked:
            style_name = (rec.get("hairstyle") or "").strip()
            normalized = normalize_style_name(style_name) if style_name else ""
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ranking.append(
                {
                    "style_name": style_name,
                    "score": round(float(rec.get("score", 0.0)) / 100.0, 2),
                }
            )
        return ranking

    def _explore_top3(
        self,
        face_shape: str,
        skin_tone: str,
        face_features: Optional[List[float]],
        skin_features: Optional[List[float]],
        gender: Optional[str],
        image_hash: str,
        experiment_arm: str,
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[int, float]], Dict[str, Any]]:
        """동점권 탐색 Top-3 (REC_EXPLORE_ENABLED 일 때만)

        실험군·대조군 모두 Top-POOL 풀을 받아 동점권 크기를 잰다. 대조군은 풀 상위
        3개를 그대로 쓴다 - 다양성 필터가 위에서부터 탐욕적으로 고르므로 k=3 호출
        결과와 같다 (v5+ 정규화 모델 기준).

        Returns:
            (표시할 ML 추천, 같은 순서의 (ml_rank, propensity), 노출 meta)
        """
        pool_size = settings.REC_EXPLORE_POOL
        # 노이즈 제외 후보보다 k 가 크면 recommend_top_k 가 전체 후보로 폴백하므로 맞춘다
        allowed = getattr(self.ml_recommender, "allowed_indices", None)
        if isinstance(allowed, list) and allowed:
            pool_size = min(pool_size, len(allowed))

        ranked = self.ml_recommender.recommend_top_k(
            face_shape=face_shape,
            skin_tone=skin_tone,
            k=max(pool_size, 3),
            face_features=face_features,
            skin_features=skin_features,
            gender=gender,
            log_selection=False,
        )

        # 띄어쓰기 중복은 _build_recommendations 가 버리므로 풀에서 미리 뺀다 (3장 유지)
        pool: List[Dict[str, Any]] = []
        seen = set()
        for rec in ranked:
            name = (rec.get("hairstyle") or "").strip()
            normalized = normalize_style_name(name) if name else ""
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            pool.append(rec)
        pool.sort(key=_raw_score, reverse=True)  # 안정 정렬 - 동점은 모델 순서 유지
        scores = [_raw_score(rec) for rec in pool]

        margin = settings.REC_EXPLORE_MARGIN
        temperature = settings.REC_EXPLORE_TEMPERATURE
        if experiment_arm == rec_exploration.ARM_EXPLORE:
            selection = rec_exploration.explore_select(
                scores,
                k=3,
                margin=margin,
                temperature=temperature,
                seed=rec_exploration.selection_seed(image_hash, self.model_version),
            )
            indices = selection.indices
            propensities = selection.propensities
            band_size = selection.band_size
            policy = rec_exploration.POLICY_EXPLORE
        else:
            indices = list(range(min(3, len(pool))))
            propensities = [1.0] * len(indices)
            band_size = rec_exploration.tie_band_size(scores, margin, 3)
            policy = rec_exploration.POLICY_DETERMINISTIC

        selected = [pool[i] for i in indices]
        logger.info(
            f"[EXPLORE] arm={experiment_arm} band={band_size}/{len(pool)} "
            f"선택: {[(r.get('hairstyle'), round(_raw_score(r), 2)) for r in selected]}"
        )
        exposure = [(i + 1, p) for i, p in zip(indices, propensities)]
        meta = {
            "rec_policy": policy,
            "experiment_arm": experiment_arm,
            "tie_band_size": band_size,
            "margin": margin,
            "temperature": temperature,
        }
        return selected, exposure, meta

    def recommend(
        self,
        image_data: bytes,
        face_shape: str,
        skin_tone: str,
        face_features: List[float] = None,
        skin_features: List[float] = None,
        gender: str = None,
        image_hash: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        ML 기반 헤어스타일 추천

        Args:
            image_data: 이미지 바이트 (현재 사용 안함, 호환성 유지)
            face_shape: 얼굴형
            skin_tone: 피부톤
            face_features: MediaPipe 얼굴 측정값 [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
            skin_features: MediaPipe 피부 측정값 [ITA_value, hue_value] (2차원)
            gender: 성별 ("male", "female", "neutral")
            image_hash: 이미지 sha256. 동점권 탐색의 시드·배정에 쓴다
                (없으면 REC_EXPLORE_ENABLED 여도 결정적 Top-3)
            user_id: 로그인 사용자 ID. 있으면 실험 배정 기준

        Returns:
            추천 결과 딕셔너리
        """
        if face_features is not None and skin_features is not None:
            logger.info(
                f"🎨 ML 추천 시작 (실제 측정값 사용): {face_shape} + {skin_tone}"
            )
        else:
            logger.info(f"🎨 ML 추천 시작 (라벨 기반): {face_shape} + {skin_tone}")
            logger.warning(
                "⚠️ 실제 측정값(face_features, skin_features)을 전달하는 것을 권장합니다."
            )

        explore_enabled = bool(settings.REC_EXPLORE_ENABLED and image_hash)
        experiment_arm = (
            rec_exploration.assign_arm(user_id, image_hash, settings.REC_EXPLORE_RATIO)
            if explore_enabled
            else None
        )
        exposure_meta: Dict[str, Any] = {
            "rec_policy": rec_exploration.POLICY_DETERMINISTIC,
            "experiment_arm": experiment_arm,
            "tie_band_size": None,
            "margin": None,
            "temperature": None,
        }
        exposure: Optional[List[Tuple[int, float]]] = None

        # ML 추천 (Top-3, 성별 필터링 적용)
        ml_recommendations = []
        if self.ml_available and self.ml_recommender and explore_enabled:
            try:
                ml_recommendations, exposure, exposure_meta = self._explore_top3(
                    face_shape,
                    skin_tone,
                    face_features,
                    skin_features,
                    gender,
                    image_hash,
                    experiment_arm,
                )
            except Exception as e:
                # 탐색 경로 오류로 추천이 비지 않도록 결정적 Top-3 로 내려간다
                logger.error(f"❌ 동점권 탐색 실패 - 결정적 Top-3 로 폴백: {str(e)}")
                explore_enabled = False
        if self.ml_available and self.ml_recommender and not explore_enabled:
            try:
                ml_recommendations = self.ml_recommender.recommend_top_k(
                    face_shape=face_shape,
                    skin_tone=skin_tone,
                    k=3,
                    face_features=face_features,
                    skin_features=skin_features,
                    gender=gender,
                )
                logger.info(f"✅ ML 추천 완료: {len(ml_recommendations)}개")
            except Exception as e:
                logger.error(f"❌ ML 추천 실패: {str(e)}")

        # 추천 결과 변환
        recommendations = self._build_recommendations(
            ml_recommendations, face_shape, skin_tone, exposure
        )

        # rank 재조정 (1, 2, 3)
        for idx, rec in enumerate(recommendations, 1):
            rec["rank"] = idx

        # 트렌드 스타일 2개 추가 (ML 추천 뒤에 배치)
        trending_count = 0
        try:
            from services.trending_style_service import get_trending_style_service

            trending_service = get_trending_style_service()
            ml_style_names = {
                normalize_style_name(r["style_name"]) for r in recommendations
            }
            trending_recs = trending_service.pick_trending(
                gender=gender or "neutral",
                exclude_styles=ml_style_names,
            )
            base_rank = len(recommendations)  # append 로 len 이 늘기 전에 고정
            for idx, tr in enumerate(trending_recs):
                tr["rank"] = base_rank + idx + 1
                recommendations.append(tr)
            trending_count = len(trending_recs)
        except Exception as e:
            logger.warning(f"트렌드 스타일 추가 실패: {e}")

        full_ranking = self._full_ranking(
            face_shape, skin_tone, face_features, skin_features, gender
        )
        if exposure_meta.get("rec_policy") == rec_exploration.POLICY_EXPLORE:
            # 탐색으로 화면 Top-3 가 점수 1~3위와 달라질 수 있으니 화면 순서에 맞춘다.
            # 결정적 경로(꺼짐/대조군)는 기존 그대로 둔다.
            full_ranking = _align_full_ranking(full_ranking, recommendations)

        # 결과 구성
        result = {
            "analysis": {
                "face_shape": face_shape,
                "personal_color": skin_tone,
                "features": "ML 모델 기반 분석",
            },
            "recommendations": recommendations,
            "full_ranking": full_ranking,
            "meta": {
                "total_count": len(recommendations),
                "ml_count": len(recommendations) - trending_count,
                "trending_count": trending_count,
                "method": "ml",
                "model_version": self.model_version,
                # 반올림 전 원점수 기준 (계측용, 기존 필드는 그대로 유지)
                **self._score_stats(ml_recommendations),
                # 노출 정책 (rec_policy / experiment_arm / tie_band_size / margin / temperature)
                **exposure_meta,
            },
        }

        logger.info(f"✅ ML 추천 완료: 총 {len(recommendations)}개")

        return result


# ========== 하위 호환성을 위한 별칭 ==========
HybridRecommendationService = MLRecommendationService


# ========== 싱글톤 인스턴스 ==========
_ml_service_instance = None


def get_ml_recommendation_service() -> MLRecommendationService:
    """
    ML 추천 서비스 싱글톤 인스턴스 가져오기

    Returns:
        MLRecommendationService 인스턴스
    """
    global _ml_service_instance

    if _ml_service_instance is None:
        logger.info("🔧 ML 추천 서비스 초기화 중...")
        _ml_service_instance = MLRecommendationService()
        logger.info("✅ ML 추천 서비스 준비 완료")

    return _ml_service_instance


def create_hybrid_service(gemini_api_key: str = None) -> MLRecommendationService:
    """
    ML 추천 서비스 인스턴스 생성 (하위 호환성)

    Args:
        gemini_api_key: 사용 안함 (하위 호환성 유지)

    Returns:
        MLRecommendationService 인스턴스
    """
    return get_ml_recommendation_service()
