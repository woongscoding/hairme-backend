"""
피드백 분석 서비스

사용자 피드백 데이터를 분석하고 통계를 제공합니다.

Author: HairMe ML Team
Date: 2025-11-13
Version: 1.0.0
"""

import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

FEEDBACK_JSON_PATH = Path("data/user_feedbacks.json")
RETRAIN_THRESHOLDS = [500, 1000, 2000, 5000]


class FeedbackAnalytics:
    """피드백 통계 분석"""

    def __init__(self):
        """초기화"""
        pass

    def _load_feedbacks(self) -> List[Dict]:
        """피드백 데이터 로드"""
        try:
            if FEEDBACK_JSON_PATH.exists():
                with open(FEEDBACK_JSON_PATH, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"❌ 피드백 로드 실패: {e}")
            return []

    def get_feedback_stats(self) -> Dict:
        """
        전체 피드백 통계 반환

        Returns:
            {
                "total": int,
                "positive_count": int,
                "negative_count": int,
                "positive_ratio": float,
                "next_retrain_threshold": int
            }
        """
        feedbacks = self._load_feedbacks()
        total = len(feedbacks)

        positive_count = sum(1 for f in feedbacks if f.get('reaction_type') == 'like')
        negative_count = sum(1 for f in feedbacks if f.get('reaction_type') == 'dislike')

        positive_ratio = round(positive_count / total * 100, 2) if total > 0 else 0.0

        # 다음 재학습 임계값 찾기
        next_threshold = None
        for threshold in RETRAIN_THRESHOLDS:
            if total < threshold:
                next_threshold = threshold
                break

        if next_threshold is None:
            next_threshold = RETRAIN_THRESHOLDS[-1]  # 마지막 임계값

        logger.info(f"📊 통계: Total={total}, Positive={positive_count}, Negative={negative_count}")

        return {
            "total": total,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "positive_ratio": positive_ratio,
            "next_retrain_threshold": next_threshold
        }

    def get_feedback_distribution(self) -> Dict:
        """
        얼굴형 및 피부톤별 피드백 분포

        Returns:
            {
                "by_face_shape": {...},
                "by_skin_tone": {...}
            }
        """
        feedbacks = self._load_feedbacks()

        # 얼굴형별 분포
        face_shape_stats = defaultdict(lambda: {"like": 0, "dislike": 0, "total": 0})

        for f in feedbacks:
            face_shape = f.get('face_shape', 'unknown')
            reaction = f.get('reaction_type', 'unknown')

            face_shape_stats[face_shape]['total'] += 1
            if reaction == 'like':
                face_shape_stats[face_shape]['like'] += 1
            elif reaction == 'dislike':
                face_shape_stats[face_shape]['dislike'] += 1

        # 피부톤별 분포
        skin_tone_stats = defaultdict(lambda: {"like": 0, "dislike": 0, "total": 0})

        for f in feedbacks:
            skin_tone = f.get('skin_tone', 'unknown')
            reaction = f.get('reaction_type', 'unknown')

            skin_tone_stats[skin_tone]['total'] += 1
            if reaction == 'like':
                skin_tone_stats[skin_tone]['like'] += 1
            elif reaction == 'dislike':
                skin_tone_stats[skin_tone]['dislike'] += 1

        # 비율 계산
        for shape, stats in face_shape_stats.items():
            if stats['total'] > 0:
                stats['like_ratio'] = round(stats['like'] / stats['total'] * 100, 2)
            else:
                stats['like_ratio'] = 0.0

        for tone, stats in skin_tone_stats.items():
            if stats['total'] > 0:
                stats['like_ratio'] = round(stats['like'] / stats['total'] * 100, 2)
            else:
                stats['like_ratio'] = 0.0

        return {
            "by_face_shape": dict(face_shape_stats),
            "by_skin_tone": dict(skin_tone_stats)
        }

    def get_top_hairstyles(self, top_n: int = 10) -> Dict:
        """
        좋아요/싫어요가 가장 많은 헤어스타일 Top N

        Args:
            top_n: 반환할 개수

        Returns:
            {
                "most_liked": [...],
                "most_disliked": [...]
            }
        """
        feedbacks = self._load_feedbacks()

        # 헤어스타일별 통계
        hairstyle_stats = defaultdict(lambda: {"like": 0, "dislike": 0, "total": 0})

        for f in feedbacks:
            hairstyle_id = f.get('hairstyle_id', -1)
            reaction = f.get('reaction_type', 'unknown')

            hairstyle_stats[hairstyle_id]['total'] += 1
            if reaction == 'like':
                hairstyle_stats[hairstyle_id]['like'] += 1
            elif reaction == 'dislike':
                hairstyle_stats[hairstyle_id]['dislike'] += 1

        # 좋아요 순으로 정렬
        most_liked = sorted(
            hairstyle_stats.items(),
            key=lambda x: x[1]['like'],
            reverse=True
        )[:top_n]

        # 싫어요 순으로 정렬
        most_disliked = sorted(
            hairstyle_stats.items(),
            key=lambda x: x[1]['dislike'],
            reverse=True
        )[:top_n]

        # 포맷 변환
        most_liked_list = [
            {
                "hairstyle_id": h_id,
                "like_count": stats['like'],
                "dislike_count": stats['dislike'],
                "total_count": stats['total']
            }
            for h_id, stats in most_liked
        ]

        most_disliked_list = [
            {
                "hairstyle_id": h_id,
                "like_count": stats['like'],
                "dislike_count": stats['dislike'],
                "total_count": stats['total']
            }
            for h_id, stats in most_disliked
        ]

        return {
            "most_liked": most_liked_list,
            "most_disliked": most_disliked_list
        }


# ========== 싱글톤 인스턴스 ==========
_analytics_instance = None


def get_feedback_analytics() -> FeedbackAnalytics:
    """
    피드백 분석 싱글톤 인스턴스

    Returns:
        FeedbackAnalytics 인스턴스
    """
    global _analytics_instance

    if _analytics_instance is None:
        logger.info("🔧 피드백 분석기 초기화 중...")
        _analytics_instance = FeedbackAnalytics()
        logger.info("✅ 피드백 분석기 준비 완료")

    return _analytics_instance
