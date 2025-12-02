"""
MLOps 서비스 모듈

AWS Lambda + DynamoDB + S3 기반의 자동 재학습 파이프라인

Components:
- s3_feedback_store: S3에 피드백 데이터 저장 (학습용 NPZ)
- training_trigger: EventBridge 기반 재학습 트리거
- ab_test: A/B 테스트 라우터 (Champion/Challenger 분배)
- ab_evaluator: A/B 테스트 성과 평가기

Author: HairMe ML Team
Date: 2025-12-02
Version: 2.0.0 (A/B 테스트 추가)
"""

from .s3_feedback_store import S3FeedbackStore, get_s3_feedback_store
from .training_trigger import TrainingTrigger, get_training_trigger
from .ab_test import (
    ABTestConfig,
    ABTestRouter,
    ModelVariant,
    get_ab_router,
    refresh_ab_router
)
from .ab_evaluator import (
    ABTestMetrics,
    ABTestEvaluator,
    get_ab_evaluator
)

__all__ = [
    # S3 피드백 저장소
    'S3FeedbackStore',
    'get_s3_feedback_store',
    # 학습 트리거
    'TrainingTrigger',
    'get_training_trigger',
    # A/B 테스트 라우터
    'ABTestConfig',
    'ABTestRouter',
    'ModelVariant',
    'get_ab_router',
    'refresh_ab_router',
    # A/B 테스트 평가기
    'ABTestMetrics',
    'ABTestEvaluator',
    'get_ab_evaluator',
]
