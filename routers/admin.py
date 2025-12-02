"""
관리자 대시보드 라우터

피드백 통계 및 MLOps 상태 API를 제공합니다.
신버전: S3 + DynamoDB 기반 MLOps 시스템

Author: HairMe ML Team
Date: 2025-12-02
Version: 2.0.0
"""

import os
from fastapi import APIRouter, HTTPException, Depends
from services.circuit_breaker import get_circuit_breaker_status, reset_circuit_breakers
from core.auth import verify_admin_api_key
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/admin/mlops-status")
async def get_mlops_status(api_key: str = Depends(verify_admin_api_key)):
    """
    MLOps 파이프라인 상태 조회

    Returns:
        - enabled: MLOps 활성화 여부
        - s3_bucket: S3 버킷 이름
        - pending_count: 대기 중인 피드백 수
        - total_feedback_count: 전체 피드백 수
        - retrain_threshold: 재학습 트리거 임계값
        - last_training_at: 마지막 학습 시간
    """
    try:
        mlops_enabled = os.getenv('MLOPS_ENABLED', 'false').lower() == 'true'

        if not mlops_enabled:
            return {
                "success": True,
                "enabled": False,
                "message": "MLOps is disabled"
            }

        # S3 피드백 저장소 통계 조회
        from services.mlops.s3_feedback_store import get_s3_feedback_store
        store = get_s3_feedback_store()
        stats = store.get_stats()

        logger.info(f"📊 MLOps 상태 조회: {stats}")

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"❌ MLOps 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"MLOps 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/feedback-stats")
async def get_feedback_stats(api_key: str = Depends(verify_admin_api_key)):
    """
    DynamoDB 기반 피드백 통계 조회

    Returns:
        - total_analysis: 전체 분석 수
        - total_feedback: 피드백이 있는 분석 수
        - like_counts: 스타일별 좋아요 수
        - dislike_counts: 스타일별 싫어요 수
    """
    try:
        use_dynamodb = os.getenv('USE_DYNAMODB', 'false').lower() == 'true'

        if use_dynamodb:
            from database.dynamodb_connection import get_feedback_stats as get_dynamodb_stats
            stats = get_dynamodb_stats()

            logger.info(f"📊 피드백 통계 조회 (DynamoDB): {stats.get('total_feedback', 0)}개")

            return stats
        else:
            return {
                "success": False,
                "message": "DynamoDB is not enabled. Set USE_DYNAMODB=true"
            }

    except Exception as e:
        logger.error(f"❌ 피드백 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"피드백 통계 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/circuit-breaker-status")
async def get_circuit_status(api_key: str = Depends(verify_admin_api_key)):
    """
    Circuit Breaker 상태 조회

    Returns:
        - gemini_api: Gemini API Circuit Breaker 상태
            - state: 현재 상태 (closed/open/half-open)
            - fail_counter: 현재 실패 횟수
            - fail_max: 최대 허용 실패 횟수
            - timeout_duration: 타임아웃 시간 (초)
            - is_open: Circuit이 Open 상태인지 여부
            - is_closed: Circuit이 Closed 상태인지 여부
            - is_half_open: Circuit이 Half-Open 상태인지 여부
    """
    try:
        status = get_circuit_breaker_status()

        logger.info(f"⚡ Circuit Breaker 상태 조회: {status}")

        return {
            "success": True,
            **status
        }

    except Exception as e:
        logger.error(f"❌ Circuit Breaker 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Circuit Breaker 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/admin/circuit-breaker-reset")
async def reset_circuit(api_key: str = Depends(verify_admin_api_key)):
    """
    Circuit Breaker 수동 리셋 (관리자 전용)

    모든 Circuit Breaker를 강제로 닫힌 상태로 리셋합니다.
    """
    try:
        reset_circuit_breakers()

        logger.warning(f"⚠️ [ADMIN] Circuit Breaker 수동 리셋 실행됨")

        return {
            "success": True,
            "message": "All circuit breakers have been reset"
        }

    except Exception as e:
        logger.error(f"❌ Circuit Breaker 리셋 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Circuit Breaker 리셋 중 오류가 발생했습니다: {str(e)}"
        )
