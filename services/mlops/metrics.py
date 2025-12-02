"""
ML 추천 시스템 평가 지표 모듈

추천 시스템의 성능을 측정하기 위한 표준 메트릭:
- Precision@K: Top-K 추천 중 적중률
- Recall@K: 전체 관련 아이템 중 Top-K에서의 적중률
- NDCG@K: 순위를 고려한 정규화된 할인 누적 이득
- MRR: 평균 역순위 (첫 번째 적중 위치)
- Hit Rate@K: Top-K 중 하나라도 맞으면 적중

Author: HairMe ML Team
Date: 2025-12-03
Version: 1.0.0
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class RecommendationMetrics:
    """추천 시스템 평가 지표 결과"""

    # 기본 지표
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    hit_rate_at_k: Dict[int, float] = field(default_factory=dict)

    # 추가 지표
    coverage: float = 0.0  # 추천된 아이템의 다양성 (전체 아이템 중 추천된 비율)
    avg_score: float = 0.0  # 평균 추천 점수

    # 메타 정보
    num_samples: int = 0
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "ndcg_at_k": self.ndcg_at_k,
            "mrr": self.mrr,
            "hit_rate_at_k": self.hit_rate_at_k,
            "coverage": self.coverage,
            "avg_score": self.avg_score,
            "num_samples": self.num_samples,
            "evaluated_at": self.evaluated_at
        }

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def summary(self) -> str:
        """요약 문자열 생성"""
        lines = [
            "=" * 50,
            "📊 ML 추천 시스템 평가 결과",
            "=" * 50,
            f"평가 샘플 수: {self.num_samples}",
            "",
            "📈 주요 지표:",
        ]

        # Precision@K
        for k, v in sorted(self.precision_at_k.items()):
            lines.append(f"  Precision@{k}: {v:.4f} ({v*100:.2f}%)")

        # NDCG@K
        for k, v in sorted(self.ndcg_at_k.items()):
            lines.append(f"  NDCG@{k}: {v:.4f}")

        # Hit Rate@K
        for k, v in sorted(self.hit_rate_at_k.items()):
            lines.append(f"  Hit Rate@{k}: {v:.4f} ({v*100:.2f}%)")

        lines.extend([
            f"  MRR: {self.mrr:.4f}",
            f"  Coverage: {self.coverage:.4f} ({self.coverage*100:.2f}%)",
            f"  평균 추천 점수: {self.avg_score:.2f}",
            "",
            f"평가 시각: {self.evaluated_at}",
            "=" * 50
        ])

        return "\n".join(lines)


class RecommendationEvaluator:
    """
    추천 시스템 평가기

    피드백 데이터를 기반으로 추천 모델의 성능을 평가합니다.

    평가 방법:
    1. 긍정 피드백 (positive): 사용자가 "좋음"으로 평가하거나 클릭한 스타일
    2. 추천 결과와 긍정 피드백 비교
    """

    def __init__(self, k_values: List[int] = None):
        """
        Args:
            k_values: 평가할 K 값 리스트 (기본값: [1, 3, 5])
        """
        self.k_values = k_values or [1, 3, 5]

    def precision_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Precision@K: Top-K 추천 중 관련 아이템 비율

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트 (정답)
            k: 상위 K개만 평가

        Returns:
            Precision@K 값 (0.0 ~ 1.0)
        """
        if not recommended or k <= 0:
            return 0.0

        top_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)

        return hits / k

    def recall_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Recall@K: 전체 관련 아이템 중 Top-K에서 적중한 비율

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트 (정답)
            k: 상위 K개만 평가

        Returns:
            Recall@K 값 (0.0 ~ 1.0)
        """
        if not relevant or k <= 0:
            return 0.0

        top_k = recommended[:k]
        relevant_set = set(relevant)
        hits = sum(1 for item in top_k if item in relevant_set)

        return hits / len(relevant)

    def dcg_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int,
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        DCG@K: 할인 누적 이득 (순위에 따른 가중치 적용)

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트
            k: 상위 K개만 평가
            relevance_scores: 아이템별 관련성 점수 (없으면 이진: 1 또는 0)

        Returns:
            DCG@K 값
        """
        if not recommended or k <= 0:
            return 0.0

        top_k = recommended[:k]
        relevant_set = set(relevant)

        dcg = 0.0
        for i, item in enumerate(top_k):
            if relevance_scores and item in relevance_scores:
                rel = relevance_scores[item]
            elif item in relevant_set:
                rel = 1.0
            else:
                rel = 0.0

            # DCG 공식: rel_i / log2(i + 2)
            dcg += rel / np.log2(i + 2)

        return dcg

    def ndcg_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int,
        relevance_scores: Dict[str, float] = None
    ) -> float:
        """
        NDCG@K: 정규화된 할인 누적 이득

        이상적인 순서 대비 실제 순서의 품질을 측정합니다.

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트
            k: 상위 K개만 평가
            relevance_scores: 아이템별 관련성 점수

        Returns:
            NDCG@K 값 (0.0 ~ 1.0)
        """
        dcg = self.dcg_at_k(recommended, relevant, k, relevance_scores)

        if dcg == 0:
            return 0.0

        # 이상적인 순서: 관련성 높은 순으로 정렬
        if relevance_scores:
            ideal_order = sorted(
                relevant,
                key=lambda x: relevance_scores.get(x, 0.0),
                reverse=True
            )
        else:
            ideal_order = relevant.copy()

        idcg = self.dcg_at_k(ideal_order, relevant, k, relevance_scores)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def mrr(
        self,
        recommended: List[str],
        relevant: List[str]
    ) -> float:
        """
        MRR (Mean Reciprocal Rank): 첫 번째 적중 위치의 역수

        단일 쿼리에 대한 RR (Reciprocal Rank)을 반환합니다.
        여러 쿼리의 평균은 evaluate() 메서드에서 계산합니다.

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트

        Returns:
            RR 값 (0.0 ~ 1.0)
        """
        if not recommended or not relevant:
            return 0.0

        relevant_set = set(relevant)

        for i, item in enumerate(recommended):
            if item in relevant_set:
                return 1.0 / (i + 1)

        return 0.0

    def hit_rate_at_k(
        self,
        recommended: List[str],
        relevant: List[str],
        k: int
    ) -> float:
        """
        Hit Rate@K: Top-K 중 하나라도 관련 아이템이 있으면 1, 없으면 0

        Args:
            recommended: 추천된 아이템 리스트 (순위순)
            relevant: 관련 아이템 리스트
            k: 상위 K개만 평가

        Returns:
            1.0 (적중) 또는 0.0 (미적중)
        """
        if not recommended or not relevant or k <= 0:
            return 0.0

        top_k = recommended[:k]
        relevant_set = set(relevant)

        for item in top_k:
            if item in relevant_set:
                return 1.0

        return 0.0

    def evaluate_single(
        self,
        recommended: List[str],
        relevant: List[str],
        relevance_scores: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        단일 추천 결과 평가

        Args:
            recommended: 추천된 아이템 리스트
            relevant: 관련 아이템 리스트 (정답)
            relevance_scores: 아이템별 관련성 점수 (선택)

        Returns:
            각 지표별 결과 딕셔너리
        """
        result = {
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {},
            "hit_rate_at_k": {},
            "rr": self.mrr(recommended, relevant)
        }

        for k in self.k_values:
            result["precision_at_k"][k] = self.precision_at_k(recommended, relevant, k)
            result["recall_at_k"][k] = self.recall_at_k(recommended, relevant, k)
            result["ndcg_at_k"][k] = self.ndcg_at_k(recommended, relevant, k, relevance_scores)
            result["hit_rate_at_k"][k] = self.hit_rate_at_k(recommended, relevant, k)

        return result

    def evaluate_batch(
        self,
        predictions: List[Tuple[List[str], List[str]]],
        relevance_scores_list: List[Dict[str, float]] = None,
        all_items: List[str] = None
    ) -> RecommendationMetrics:
        """
        여러 추천 결과를 배치로 평가

        Args:
            predictions: [(추천 리스트, 정답 리스트), ...] 형태의 리스트
            relevance_scores_list: 각 예측에 대한 관련성 점수 리스트 (선택)
            all_items: 전체 아이템 리스트 (coverage 계산용)

        Returns:
            RecommendationMetrics 객체
        """
        if not predictions:
            return RecommendationMetrics()

        n = len(predictions)

        # 각 지표 누적
        precision_sums = {k: 0.0 for k in self.k_values}
        recall_sums = {k: 0.0 for k in self.k_values}
        ndcg_sums = {k: 0.0 for k in self.k_values}
        hit_rate_sums = {k: 0.0 for k in self.k_values}
        mrr_sum = 0.0
        score_sum = 0.0
        score_count = 0

        # 추천된 아이템 추적 (coverage 계산용)
        recommended_items = set()

        for i, (recommended, relevant) in enumerate(predictions):
            rel_scores = relevance_scores_list[i] if relevance_scores_list else None

            # 단일 평가
            single_result = self.evaluate_single(recommended, relevant, rel_scores)

            # 누적
            for k in self.k_values:
                precision_sums[k] += single_result["precision_at_k"][k]
                recall_sums[k] += single_result["recall_at_k"][k]
                ndcg_sums[k] += single_result["ndcg_at_k"][k]
                hit_rate_sums[k] += single_result["hit_rate_at_k"][k]

            mrr_sum += single_result["rr"]

            # 추천 아이템 추적
            recommended_items.update(recommended)

            # 관련성 점수 누적
            if rel_scores:
                for item in recommended:
                    if item in rel_scores:
                        score_sum += rel_scores[item]
                        score_count += 1

        # 평균 계산
        metrics = RecommendationMetrics(
            precision_at_k={k: v / n for k, v in precision_sums.items()},
            recall_at_k={k: v / n for k, v in recall_sums.items()},
            ndcg_at_k={k: v / n for k, v in ndcg_sums.items()},
            hit_rate_at_k={k: v / n for k, v in hit_rate_sums.items()},
            mrr=mrr_sum / n,
            num_samples=n,
            avg_score=score_sum / score_count if score_count > 0 else 0.0
        )

        # Coverage 계산
        if all_items:
            metrics.coverage = len(recommended_items) / len(all_items)

        return metrics


class FeedbackEvaluator:
    """
    피드백 기반 평가기

    실제 사용자 피드백 데이터를 기반으로 모델 성능을 평가합니다.
    """

    def __init__(
        self,
        positive_threshold: float = 70.0,
        k_values: List[int] = None
    ):
        """
        Args:
            positive_threshold: 긍정 피드백으로 간주할 점수 임계값
            k_values: 평가할 K 값 리스트
        """
        self.positive_threshold = positive_threshold
        self.evaluator = RecommendationEvaluator(k_values or [1, 3, 5])

    def evaluate_from_feedback(
        self,
        feedback_data: List[Dict[str, Any]],
        model_predictions: List[List[str]] = None,
        all_styles: List[str] = None
    ) -> RecommendationMetrics:
        """
        피드백 데이터로부터 평가

        Args:
            feedback_data: 피드백 데이터 리스트
                [{
                    "recommended_styles": ["스타일1", "스타일2", "스타일3"],
                    "selected_style": "스타일1",  # 사용자가 선택한 스타일
                    "feedback": "good" | "bad",
                    "score": 90  # 피드백 점수 (선택)
                }, ...]
            model_predictions: 모델 예측 리스트 (없으면 feedback_data에서 추출)
            all_styles: 전체 스타일 리스트 (coverage 계산용)

        Returns:
            RecommendationMetrics 객체
        """
        predictions = []
        relevance_scores_list = []

        for item in feedback_data:
            # 추천 결과
            recommended = item.get("recommended_styles", [])

            # 정답 (긍정 피드백을 받은 스타일)
            relevant = []
            relevance_scores = {}

            selected_style = item.get("selected_style")
            feedback = item.get("feedback", "").lower()
            score = item.get("score", 0)

            # 긍정 피드백인 경우 해당 스타일을 정답으로
            if feedback == "good" or score >= self.positive_threshold:
                if selected_style:
                    relevant.append(selected_style)
                    # 정규화된 관련성 점수 (0~1)
                    relevance_scores[selected_style] = score / 100.0 if score else 0.9

            # 추천이 있고 정답도 있는 경우만 평가
            if recommended and relevant:
                predictions.append((recommended, relevant))
                relevance_scores_list.append(relevance_scores)

        if not predictions:
            logger.warning("⚠️ 평가 가능한 피드백 데이터가 없습니다.")
            return RecommendationMetrics()

        logger.info(f"📊 {len(predictions)}개 샘플로 평가 수행")

        return self.evaluator.evaluate_batch(
            predictions,
            relevance_scores_list,
            all_styles
        )

    def calculate_business_metrics(
        self,
        feedback_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        비즈니스 지표 계산

        Args:
            feedback_data: 피드백 데이터 리스트

        Returns:
            비즈니스 지표 딕셔너리
        """
        total = len(feedback_data)
        if total == 0:
            return {
                "total_feedback": 0,
                "positive_rate": 0.0,
                "negative_rate": 0.0,
                "click_rate": 0.0,
                "avg_score": 0.0
            }

        positive_count = 0
        negative_count = 0
        click_count = 0
        score_sum = 0.0
        score_count = 0

        for item in feedback_data:
            feedback = item.get("feedback", "").lower()
            score = item.get("score", 0)
            clicked = item.get("naver_clicked", False)

            if feedback == "good" or score >= self.positive_threshold:
                positive_count += 1
            elif feedback == "bad" or (score > 0 and score < self.positive_threshold):
                negative_count += 1

            if clicked:
                click_count += 1

            if score > 0:
                score_sum += score
                score_count += 1

        return {
            "total_feedback": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_rate": positive_count / total,
            "negative_rate": negative_count / total,
            "click_rate": click_count / total,
            "avg_score": score_sum / score_count if score_count > 0 else 0.0,
            "evaluated_at": datetime.utcnow().isoformat()
        }


# ========== 편의 함수 ==========

def evaluate_model_performance(
    predictions: List[Tuple[List[str], List[str]]],
    k_values: List[int] = None
) -> RecommendationMetrics:
    """
    모델 성능 평가 (편의 함수)

    Args:
        predictions: [(추천 리스트, 정답 리스트), ...]
        k_values: 평가할 K 값 리스트

    Returns:
        RecommendationMetrics 객체
    """
    evaluator = RecommendationEvaluator(k_values or [1, 3, 5])
    return evaluator.evaluate_batch(predictions)


def evaluate_from_s3_feedback(
    s3_bucket: str,
    feedback_prefix: str = "feedback/processed/",
    positive_threshold: float = 70.0
) -> RecommendationMetrics:
    """
    S3에 저장된 피드백으로 평가 (편의 함수)

    Args:
        s3_bucket: S3 버킷 이름
        feedback_prefix: 피드백 파일 prefix
        positive_threshold: 긍정 피드백 임계값

    Returns:
        RecommendationMetrics 객체
    """
    try:
        import boto3
        import numpy as np

        s3 = boto3.client('s3')

        # 피드백 파일 리스트
        response = s3.list_objects_v2(
            Bucket=s3_bucket,
            Prefix=feedback_prefix
        )

        feedback_data = []

        for obj in response.get('Contents', []):
            key = obj['Key']
            if key.endswith('.npz'):
                # NPZ 파일 로드
                result = s3.get_object(Bucket=s3_bucket, Key=key)
                data = np.load(result['Body'], allow_pickle=True)

                # ground_truth를 점수로 사용
                if 'ground_truth' in data:
                    score = float(data['ground_truth'])
                    feedback_data.append({
                        "score": score,
                        "feedback": "good" if score >= positive_threshold else "bad"
                    })

        if not feedback_data:
            logger.warning(f"⚠️ {feedback_prefix}에서 피드백 데이터를 찾을 수 없습니다.")
            return RecommendationMetrics()

        evaluator = FeedbackEvaluator(positive_threshold)
        return evaluator.calculate_business_metrics(feedback_data)

    except Exception as e:
        logger.error(f"❌ S3 피드백 평가 실패: {e}")
        return RecommendationMetrics()


# ========== 테스트용 ==========

if __name__ == "__main__":
    # 테스트 데이터
    test_predictions = [
        (["스타일A", "스타일B", "스타일C"], ["스타일A"]),
        (["스타일D", "스타일E", "스타일F"], ["스타일E"]),
        (["스타일G", "스타일H", "스타일I"], ["스타일J"]),  # Miss
        (["스타일K", "스타일L", "스타일M"], ["스타일K", "스타일L"]),  # 2개 적중
    ]

    # 평가 실행
    evaluator = RecommendationEvaluator(k_values=[1, 3])
    metrics = evaluator.evaluate_batch(test_predictions)

    print(metrics.summary())
    print("\n📋 JSON 출력:")
    print(metrics.to_json())
