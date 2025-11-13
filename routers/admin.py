"""
관리자 대시보드 라우터

피드백 통계 및 분석 API를 제공합니다.

Author: HairMe ML Team
Date: 2025-11-13
Version: 1.0.0
"""

from fastapi import APIRouter, HTTPException
from services.feedback_analytics import get_feedback_analytics
from services.retrain_queue import get_retrain_queue
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/admin/feedback-stats")
async def get_feedback_stats():
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
async def get_feedback_distribution():
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
async def get_top_hairstyles(top_n: int = 10):
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
async def get_retrain_status():
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
