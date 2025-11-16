"""
관리자 대시보드 라우터

피드백 통계 및 분석 API를 제공합니다.

Author: HairMe ML Team
Date: 2025-11-13
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException, Depends
from services.feedback_analytics import get_feedback_analytics
from services.retrain_queue import get_retrain_queue
from services.circuit_breaker import get_circuit_breaker_status, reset_circuit_breakers
from core.auth import verify_admin_api_key
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/admin/feedback-stats")
async def get_feedback_stats(api_key: str = Depends(verify_admin_api_key)):
    """
    전체 피드백 통계 조회

    Returns:
        - total: 전체 피드백 수
        - positive_count: 좋아요 수
        - negative_count: 싫어요 수
        - positive_ratio: 좋아요 비율 (%)
        - next_retrain_threshold: 다음 재학습 임계값
    """
    try:
        analytics = get_feedback_analytics()
        stats = analytics.get_feedback_stats()

        logger.info(f"📊 통계 조회 성공: {stats['total']}개 피드백")

        return {
            "success": True,
            **stats
        }

    except Exception as e:
        logger.error(f"❌ 통계 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"통계 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/feedback-distribution")
async def get_feedback_distribution(api_key: str = Depends(verify_admin_api_key)):
    """
    얼굴형 및 피부톤별 피드백 분포

    Returns:
        - by_face_shape: 얼굴형별 통계
        - by_skin_tone: 피부톤별 통계
    """
    try:
        analytics = get_feedback_analytics()
        distribution = analytics.get_feedback_distribution()

        logger.info(f"📊 분포 조회 성공")

        return {
            "success": True,
            **distribution
        }

    except Exception as e:
        logger.error(f"❌ 분포 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"분포 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/top-hairstyles")
async def get_top_hairstyles(top_n: int = 10, api_key: str = Depends(verify_admin_api_key)):
    """
    좋아요/싫어요가 많은 헤어스타일 Top N

    Args:
        top_n: 반환할 개수 (기본값: 10)

    Returns:
        - most_liked: 좋아요가 많은 스타일 리스트
        - most_disliked: 싫어요가 많은 스타일 리스트
    """
    try:
        analytics = get_feedback_analytics()
        top_styles = analytics.get_top_hairstyles(top_n=top_n)

        logger.info(f"📊 Top {top_n} 헤어스타일 조회 성공")

        return {
            "success": True,
            **top_styles
        }

    except Exception as e:
        logger.error(f"❌ Top 헤어스타일 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Top 헤어스타일 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/retrain-status")
async def get_retrain_status(api_key: str = Depends(verify_admin_api_key)):
    """
    재학습 작업 상태 조회

    Returns:
        - queue_stats: 큐 통계 (total, pending, running, completed, failed)
        - pending_jobs: 대기 중인 작업 리스트
        - recent_jobs: 최근 5개 작업 리스트
    """
    try:
        queue = get_retrain_queue()

        # 큐 통계
        stats = queue.get_queue_stats()

        # 대기 중인 작업
        pending_jobs = queue.get_pending_jobs()

        # 최근 5개 작업
        all_jobs = queue.get_all_jobs()
        all_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        recent_jobs = all_jobs[:5]

        logger.info(f"🔄 재학습 큐 상태 조회 성공: {stats}")

        return {
            "success": True,
            "queue_stats": stats,
            "pending_jobs": pending_jobs,
            "recent_jobs": recent_jobs
        }

    except Exception as e:
        logger.error(f"❌ 재학습 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"재학습 상태 조회 중 오류가 발생했습니다: {str(e)}"
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
