"""
MLOps 서비스 모듈

AWS Lambda + DynamoDB + S3 기반의 자동 재학습 파이프라인

Components:
- s3_feedback_store: S3에 피드백 데이터 저장 (학습용 NPZ)
- training_trigger: EventBridge 기반 재학습 트리거

모델 승격 판정은 lambda_trainer 의 홀드아웃 품질 게이트가 담당한다
(A/B 테스트 라우터/평가기는 서빙 경로에서 호출된 적이 없어 제거됨).

Author: HairMe ML Team
Date: 2025-12-02
Version: 3.0.0 (A/B 테스트 제거, 홀드아웃 게이트로 대체)
"""

from .s3_feedback_store import S3FeedbackStore, get_s3_feedback_store
from .training_trigger import TrainingTrigger, get_training_trigger

__all__ = [
    # S3 피드백 저장소
    "S3FeedbackStore",
    "get_s3_feedback_store",
    # 학습 트리거
    "TrainingTrigger",
    "get_training_trigger",
]
