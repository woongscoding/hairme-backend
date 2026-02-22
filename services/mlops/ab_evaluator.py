"""
A/B 테스트 평가기 모듈

DynamoDB에서 실험 데이터를 조회하여 Champion과 Challenger 모델의
성능을 비교 분석합니다.

핵심 지표:
- positive_feedback_rate: good / (good + bad)
- score_discrimination: 좋은 스타일과 나쁜 스타일의 점수 차이

Author: HairMe ML Team
Date: 2025-12-02
Version: 1.0.0
"""

import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ABTestMetrics:
    """
    A/B 테스트 변형별 지표

    Attributes:
        variant: "champion" or "challenger"
        sample_count: 총 피드백 수
        positive_count: good 피드백 수
        negative_count: bad 피드백 수
        positive_feedback_rate: good / (good + bad)
        avg_score_for_good: good 받은 스타일의 평균 예측 점수
        avg_score_for_bad: bad 받은 스타일의 평균 예측 점수
        score_discrimination: avg_good - avg_bad (클수록 좋음)
    """

    variant: str
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    positive_feedback_rate: float = 0.0
    avg_score_for_good: float = 0.0
    avg_score_for_bad: float = 0.0
    score_discrimination: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "variant": self.variant,
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_feedback_rate": round(self.positive_feedback_rate, 4),
            "avg_score_for_good": round(self.avg_score_for_good, 2),
            "avg_score_for_bad": round(self.avg_score_for_bad, 2),
            "score_discrimination": round(self.score_discrimination, 2),
        }


class ABTestEvaluator:
    """
    A/B 테스트 평가기

    DynamoDB에서 실험 데이터를 조회하고 변형별 지표를 계산합니다.

    Usage:
        evaluator = ABTestEvaluator()
        metrics = evaluator.get_metrics_by_variant("exp_2025_12_02")
        result = evaluator.is_challenger_better(metrics)
    """

    def __init__(self):
        """초기화"""
        self.dynamodb_table = None
        self._init_dynamodb()

    def _init_dynamodb(self) -> bool:
        """DynamoDB 테이블 초기화"""
        use_dynamodb = os.getenv("USE_DYNAMODB", "false").lower() == "true"
        if not use_dynamodb:
            logger.warning("⚠️ DynamoDB가 활성화되지 않음 - A/B 테스트 평가 불가")
            return False

        try:
            import boto3
            from botocore.config import Config

            aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
            table_name = os.getenv("DYNAMODB_TABLE_NAME", "hairme-analysis")

            config = Config(
                connect_timeout=5, read_timeout=10, retries={"max_attempts": 3}
            )

            dynamodb = boto3.resource("dynamodb", region_name=aws_region, config=config)
            self.dynamodb_table = dynamodb.Table(table_name)
            return True

        except Exception as e:
            logger.error(f"❌ DynamoDB 초기화 실패: {e}")
            return False

    def get_metrics_by_variant(
        self, experiment_id: str, limit: int = 10000
    ) -> Dict[str, ABTestMetrics]:
        """
        실험 ID로 변형별 지표 계산

        DynamoDB에서 해당 실험의 피드백 데이터를 조회하고
        Champion과 Challenger 각각의 지표를 계산합니다.

        Args:
            experiment_id: 실험 ID (예: "exp_2025_12_02")
            limit: 최대 조회 건수

        Returns:
            {
                'champion': ABTestMetrics(...),
                'challenger': ABTestMetrics(...)
            }
        """
        if not self.dynamodb_table:
            logger.error("❌ DynamoDB 테이블이 초기화되지 않음")
            return {}

        try:
            # DynamoDB에서 실험 데이터 조회
            # Note: 실제 운영에서는 GSI (experiment_id-feedback_at-index)를 사용해야 함
            response = self.dynamodb_table.scan(
                FilterExpression="experiment_id = :exp_id AND attribute_exists(feedback_at)",
                ExpressionAttributeValues={":exp_id": experiment_id},
                Limit=limit,
            )

            items = response.get("Items", [])

            # 페이지네이션 처리
            while "LastEvaluatedKey" in response and len(items) < limit:
                response = self.dynamodb_table.scan(
                    FilterExpression="experiment_id = :exp_id AND attribute_exists(feedback_at)",
                    ExpressionAttributeValues={":exp_id": experiment_id},
                    ExclusiveStartKey=response["LastEvaluatedKey"],
                    Limit=limit - len(items),
                )
                items.extend(response.get("Items", []))

            logger.info(
                f"📊 A/B 테스트 데이터 조회: experiment={experiment_id}, count={len(items)}"
            )

            # 변형별로 데이터 분류
            champion_data = []
            challenger_data = []

            for item in items:
                variant = item.get("ab_variant", "champion")
                if variant == "challenger":
                    challenger_data.append(item)
                else:
                    champion_data.append(item)

            # 각 변형별 지표 계산
            champion_metrics = self._calculate_metrics("champion", champion_data)
            challenger_metrics = self._calculate_metrics("challenger", challenger_data)

            return {"champion": champion_metrics, "challenger": challenger_metrics}

        except Exception as e:
            logger.error(f"❌ A/B 테스트 데이터 조회 실패: {e}")
            return {}

    def _calculate_metrics(self, variant: str, items: List[Dict]) -> ABTestMetrics:
        """
        단일 변형에 대한 지표 계산

        Args:
            variant: "champion" or "challenger"
            items: DynamoDB 아이템 리스트

        Returns:
            ABTestMetrics 인스턴스
        """
        metrics = ABTestMetrics(variant=variant)

        if not items:
            return metrics

        good_scores = []
        bad_scores = []

        for item in items:
            # 3개 스타일 각각에 대해 피드백 확인
            for i in range(1, 4):
                feedback_key = f"style_{i}_feedback"
                score_key = f"style_{i}_score"

                feedback = item.get(feedback_key)
                score = item.get(score_key)

                if feedback in ["good", "like"]:
                    metrics.positive_count += 1
                    if score is not None:
                        good_scores.append(float(score))
                elif feedback in ["bad", "dislike"]:
                    metrics.negative_count += 1
                    if score is not None:
                        bad_scores.append(float(score))

        # 총 샘플 수
        metrics.sample_count = metrics.positive_count + metrics.negative_count

        # 긍정 피드백 비율
        if metrics.sample_count > 0:
            metrics.positive_feedback_rate = (
                metrics.positive_count / metrics.sample_count
            )

        # 평균 점수 계산
        if good_scores:
            metrics.avg_score_for_good = sum(good_scores) / len(good_scores)

        if bad_scores:
            metrics.avg_score_for_bad = sum(bad_scores) / len(bad_scores)

        # 점수 구분력 (좋은 스타일과 나쁜 스타일의 점수 차이)
        if good_scores and bad_scores:
            metrics.score_discrimination = (
                metrics.avg_score_for_good - metrics.avg_score_for_bad
            )

        return metrics

    def is_challenger_better(
        self,
        metrics: Dict[str, ABTestMetrics],
        min_samples: int = 100,
        min_improvement: float = 0.02,
    ) -> Dict[str, Any]:
        """
        Challenger가 Champion보다 나은지 판단

        Args:
            metrics: get_metrics_by_variant() 반환값
            min_samples: 최소 필요 샘플 수
            min_improvement: 최소 개선율 (기본 2%)

        Returns:
            {
                "conclusion": "challenger_wins" | "champion_wins" | "no_difference" | "insufficient_data",
                "champion_metrics": {...},
                "challenger_metrics": {...},
                "improvement": 0.05,  # 5% 개선
                "recommendation": "새 모델로 교체를 권장합니다" | "기존 모델 유지 권장",
                "confidence": "high" | "medium" | "low"
            }
        """
        champion = metrics.get("champion")
        challenger = metrics.get("challenger")

        result = {
            "champion_metrics": champion.to_dict() if champion else None,
            "challenger_metrics": challenger.to_dict() if challenger else None,
            "conclusion": "insufficient_data",
            "improvement": 0.0,
            "recommendation": "",
            "confidence": "low",
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }

        # 데이터 검증
        if not champion or not challenger:
            result["recommendation"] = (
                "변형별 데이터가 부족합니다. 더 많은 피드백을 수집해주세요."
            )
            return result

        # 최소 샘플 수 확인
        if champion.sample_count < min_samples or challenger.sample_count < min_samples:
            result["recommendation"] = (
                f"최소 샘플 수({min_samples})를 충족하지 않습니다. "
                f"Champion: {champion.sample_count}, Challenger: {challenger.sample_count}"
            )
            return result

        # 긍정 피드백 비율 비교
        rate_diff = challenger.positive_feedback_rate - champion.positive_feedback_rate
        result["improvement"] = round(rate_diff, 4)

        # 신뢰도 계산 (샘플 수 기반)
        total_samples = champion.sample_count + challenger.sample_count
        if total_samples >= 1000:
            result["confidence"] = "high"
        elif total_samples >= 500:
            result["confidence"] = "medium"
        else:
            result["confidence"] = "low"

        # 결론 도출
        if rate_diff > min_improvement:
            result["conclusion"] = "challenger_wins"
            result["recommendation"] = (
                f"Challenger 모델이 {rate_diff*100:.1f}% 더 높은 긍정 피드백 비율을 보입니다. "
                f"새 모델로 교체를 권장합니다."
            )
        elif rate_diff < -min_improvement:
            result["conclusion"] = "champion_wins"
            result["recommendation"] = (
                f"Champion 모델이 {-rate_diff*100:.1f}% 더 높은 긍정 피드백 비율을 보입니다. "
                f"기존 모델 유지를 권장합니다."
            )
        else:
            result["conclusion"] = "no_difference"
            result["recommendation"] = (
                f"두 모델 간 유의미한 차이가 없습니다 (차이: {rate_diff*100:.1f}%). "
                f"더 많은 데이터를 수집하거나 실험을 종료하세요."
            )

        # 점수 구분력도 함께 고려
        if challenger.score_discrimination > champion.score_discrimination:
            result["recommendation"] += " Challenger의 점수 구분력이 더 좋습니다."
        elif challenger.score_discrimination < champion.score_discrimination:
            result["recommendation"] += " Champion의 점수 구분력이 더 좋습니다."

        logger.info(
            f"📊 A/B 테스트 평가 완료: {result['conclusion']} "
            f"(improvement={result['improvement']:.2%}, confidence={result['confidence']})"
        )

        return result

    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        실험 요약 정보 반환

        Args:
            experiment_id: 실험 ID

        Returns:
            {
                'experiment_id': 'exp_2025_12_02',
                'status': 'running' | 'completed',
                'started_at': '2025-12-02T00:00:00Z',
                'duration_days': 3,
                'total_samples': 250,
                'champion_samples': 225,
                'challenger_samples': 25,
                'current_winner': 'challenger'
            }
        """
        metrics = self.get_metrics_by_variant(experiment_id)

        champion = metrics.get("champion", ABTestMetrics(variant="champion"))
        challenger = metrics.get("challenger", ABTestMetrics(variant="challenger"))

        total = champion.sample_count + challenger.sample_count

        # 현재 승자 판단 (단순 비율 비교)
        if total < 50:
            current_winner = "undetermined"
        elif challenger.positive_feedback_rate > champion.positive_feedback_rate:
            current_winner = "challenger"
        elif champion.positive_feedback_rate > challenger.positive_feedback_rate:
            current_winner = "champion"
        else:
            current_winner = "tie"

        return {
            "experiment_id": experiment_id,
            "status": "running" if total < 100 else "analyzing",
            "total_samples": total,
            "champion_samples": champion.sample_count,
            "challenger_samples": challenger.sample_count,
            "champion_positive_rate": round(champion.positive_feedback_rate, 4),
            "challenger_positive_rate": round(challenger.positive_feedback_rate, 4),
            "current_winner": current_winner,
        }


# ========== 싱글톤 인스턴스 ==========
_evaluator_instance: Optional[ABTestEvaluator] = None


def get_ab_evaluator() -> ABTestEvaluator:
    """
    A/B 테스트 평가기 싱글톤 인스턴스 반환

    Returns:
        ABTestEvaluator 인스턴스
    """
    global _evaluator_instance

    if _evaluator_instance is None:
        _evaluator_instance = ABTestEvaluator()

    return _evaluator_instance
