"""
관리자 대시보드 라우터

피드백 통계, MLOps 상태, 크레딧 관리 API를 제공합니다.
S3 + DynamoDB 기반 MLOps 시스템 (재학습은 홀드아웃 품질 게이트로 자동 승격 판정).

Author: HairMe ML Team
Date: 2025-12-02
Version: 4.0.0
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from services.circuit_breaker import get_circuit_breaker_status, reset_circuit_breakers
from core.auth import verify_admin_api_key
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/admin/mlops-status")
@limiter.limit("10/minute")
async def get_mlops_status(
    request: Request, api_key: str = Depends(verify_admin_api_key)
):
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
        mlops_enabled = os.getenv("MLOPS_ENABLED", "false").lower() == "true"

        if not mlops_enabled:
            return {"success": True, "enabled": False, "message": "MLOps is disabled"}

        # S3 피드백 저장소 통계 조회
        from services.mlops.s3_feedback_store import get_s3_feedback_store

        store = get_s3_feedback_store()
        stats = store.get_stats()

        logger.info(f"📊 MLOps 상태 조회: {stats}")

        return {"success": True, **stats}

    except Exception as e:
        logger.error(f"❌ MLOps 상태 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )


@router.get("/admin/feedback-stats")
@limiter.limit("10/minute")
async def get_feedback_stats(
    request: Request, api_key: str = Depends(verify_admin_api_key)
):
    """
    DynamoDB 기반 피드백 통계 조회

    Returns:
        - total_analysis: 전체 분석 수
        - total_feedback: 피드백이 있는 분석 수
        - like_counts: 스타일별 좋아요 수
        - dislike_counts: 스타일별 싫어요 수
    """
    try:
        use_dynamodb = os.getenv("USE_DYNAMODB", "false").lower() == "true"

        if use_dynamodb:
            from database.dynamodb_connection import (
                get_feedback_stats as get_dynamodb_stats,
            )

            stats = get_dynamodb_stats()

            logger.info(
                f"📊 피드백 통계 조회 (DynamoDB): {stats.get('total_feedback', 0)}개"
            )

            return stats
        else:
            return {
                "success": False,
                "message": "DynamoDB is not enabled. Set USE_DYNAMODB=true",
            }

    except Exception as e:
        logger.error(f"❌ 피드백 통계 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )


@router.get("/admin/circuit-breaker-status")
@limiter.limit("10/minute")
async def get_circuit_status(
    request: Request, api_key: str = Depends(verify_admin_api_key)
):
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

        return {"success": True, **status}

    except Exception as e:
        logger.error(f"❌ Circuit Breaker 상태 조회 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )


@router.post("/admin/circuit-breaker-reset")
@limiter.limit("5/minute")
async def reset_circuit(request: Request, api_key: str = Depends(verify_admin_api_key)):
    """
    Circuit Breaker 수동 리셋 (관리자 전용)

    모든 Circuit Breaker를 강제로 닫힌 상태로 리셋합니다.
    """
    try:
        reset_circuit_breakers()

        logger.warning(f"⚠️ [ADMIN] Circuit Breaker 수동 리셋 실행됨")

        return {"success": True, "message": "All circuit breakers have been reset"}

    except Exception as e:
        logger.error(f"❌ Circuit Breaker 리셋 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )


# ========== 크레딧 관리 (관리자) ==========


class CreditGrantRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="지급 대상 user_id")
    amount: int = Field(..., ge=1, le=1000, description="지급 크레딧 수")
    memo: str = Field("", max_length=200, description="지급 사유 메모")


@router.post("/admin/credits/grant")
@limiter.limit("30/minute")
async def grant_credits(
    request: Request,
    body: CreditGrantRequest,
    api_key: str = Depends(verify_admin_api_key),
):
    """관리자 수동 크레딧 지급 (CS 보상, 이벤트 등)"""
    from services.credit_service import get_credit_service

    try:
        balance = get_credit_service().grant(
            body.user_id,
            body.amount,
            reason="admin_grant",
            ref_id=body.memo or None,
        )
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 관리자 크레딧 지급 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"👑 관리자 크레딧 지급: user_id={body.user_id}, +{body.amount} ({body.memo})"
    )
    return {
        "success": True,
        "user_id": body.user_id,
        "granted": body.amount,
        "balance": balance,
    }


# ========== 계정 관리 (관리자) ==========


@router.post("/admin/users/{user_id}/suspend")
@limiter.limit("10/minute")
async def suspend_user(
    request: Request,
    user_id: str,
    api_key: str = Depends(verify_admin_api_key),
) -> Dict[str, Any]:
    """계정 정지 (status=suspended + token_version 증가)

    - 리프레시 토큰이 전부 무효화되어 재로그인/토큰 갱신이 막힌다 (401)
    - 카카오 로그인도 403 으로 거부된다
    - 이미 발급된 액세스 토큰은 만료(기본 60분)까지 유효하다
    """
    from database.user_repository import STATUS_SUSPENDED, get_user_repository

    repo = get_user_repository()
    try:
        repo.set_status(user_id, STATUS_SUSPENDED)
        new_version = repo.bump_token_version(user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 계정 정지 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.warning(f"⛔ [ADMIN] 계정 정지: user_id={user_id}")
    return {
        "success": True,
        "user_id": user_id,
        "status": STATUS_SUSPENDED,
        "token_version": new_version,
    }


@router.post("/admin/users/{user_id}/reactivate")
@limiter.limit("10/minute")
async def reactivate_user(
    request: Request,
    user_id: str,
    api_key: str = Depends(verify_admin_api_key),
) -> Dict[str, Any]:
    """계정 정지 해제 (status=active)

    token_version 은 되돌리지 않는다 (정지 중 무효화된 세션은 그대로 유지).
    사용자는 다시 카카오 로그인해야 한다.
    """
    from database.user_repository import STATUS_ACTIVE, get_user_repository

    try:
        get_user_repository().set_status(user_id, STATUS_ACTIVE)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 계정 정지 해제 실패: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.warning(f"✅ [ADMIN] 계정 정지 해제: user_id={user_id}")
    return {"success": True, "user_id": user_id, "status": STATUS_ACTIVE}
