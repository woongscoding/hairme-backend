"""
재학습 작업 큐 관리

재학습이 필요한 시점에 작업을 큐에 추가하고 상태를 관리합니다.

Author: HairMe ML Team
Date: 2025-11-13
Version: 1.0.0
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

RETRAIN_QUEUE_PATH = Path("data/retrain_queue.json")
RETRAIN_THRESHOLDS = [500, 1000, 2000, 5000]


class RetrainQueue:
    """재학습 작업 큐 관리"""

    def __init__(self):
        """초기화"""
        # data 디렉토리 생성
        RETRAIN_QUEUE_PATH.parent.mkdir(exist_ok=True)

        # 큐 파일 초기화
        if not RETRAIN_QUEUE_PATH.exists():
            self._save_queue([])

    def _load_queue(self) -> List[Dict]:
        """큐 데이터 로드"""
        try:
            with open(RETRAIN_QUEUE_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 큐 로드 실패: {e}")
            return []

    def _save_queue(self, queue: List[Dict]):
        """큐 데이터 저장"""
        try:
            with open(RETRAIN_QUEUE_PATH, 'w', encoding='utf-8') as f:
                json.dump(queue, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ 큐 저장 실패: {e}")
            raise

    def add_job(self, feedback_count: int) -> Dict:
        """
        재학습 작업 추가

        Args:
            feedback_count: 현재 피드백 개수

        Returns:
            작업 정보
        """
        timestamp = datetime.now().isoformat()
        job_id = f"retrain_{feedback_count}_{timestamp.replace(':', '-').replace('.', '-')}"

        job = {
            "job_id": job_id,
            "feedback_count": feedback_count,
            "status": "pending",
            "created_at": timestamp,
            "started_at": None,
            "completed_at": None,
            "error_message": None
        }

        # 큐에 추가
        queue = self._load_queue()
        queue.append(job)
        self._save_queue(queue)

        logger.info(f"✅ 재학습 작업 추가: {job_id} (피드백 {feedback_count}개)")

        return job

    def get_all_jobs(self) -> List[Dict]:
        """모든 작업 조회"""
        return self._load_queue()

    def get_pending_jobs(self) -> List[Dict]:
        """대기 중인 작업 조회"""
        queue = self._load_queue()
        return [job for job in queue if job.get('status') == 'pending']

    def get_job(self, job_id: str) -> Optional[Dict]:
        """특정 작업 조회"""
        queue = self._load_queue()
        for job in queue:
            if job.get('job_id') == job_id:
                return job
        return None

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error_message: Optional[str] = None
    ):
        """
        작업 상태 업데이트

        Args:
            job_id: 작업 ID
            status: "pending", "running", "completed", "failed"
            error_message: 에러 메시지 (실패 시)
        """
        queue = self._load_queue()

        for job in queue:
            if job.get('job_id') == job_id:
                job['status'] = status

                if status == "running" and job.get('started_at') is None:
                    job['started_at'] = datetime.now().isoformat()
                elif status in ["completed", "failed"]:
                    job['completed_at'] = datetime.now().isoformat()

                if error_message:
                    job['error_message'] = error_message

                break

        self._save_queue(queue)
        logger.info(f"✅ 작업 상태 업데이트: {job_id} → {status}")

    def get_queue_stats(self) -> Dict:
        """큐 통계"""
        queue = self._load_queue()

        stats = {
            "total": len(queue),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0
        }

        for job in queue:
            status = job.get('status', 'unknown')
            if status in stats:
                stats[status] += 1

        return stats

    def clear_completed_jobs(self, keep_last_n: int = 10):
        """
        완료된 작업 정리 (최근 N개만 유지)

        Args:
            keep_last_n: 유지할 완료 작업 개수
        """
        queue = self._load_queue()

        # 완료/실패 작업과 대기/진행중 작업 분리
        completed_jobs = [j for j in queue if j.get('status') in ['completed', 'failed']]
        active_jobs = [j for j in queue if j.get('status') in ['pending', 'running']]

        # 완료 작업은 최근 N개만 유지
        completed_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        completed_jobs = completed_jobs[:keep_last_n]

        # 병합
        new_queue = active_jobs + completed_jobs
        self._save_queue(new_queue)

        logger.info(f"✅ 큐 정리 완료: {len(queue)} → {len(new_queue)}개")


# ========== 싱글톤 인스턴스 ==========
_queue_instance = None


def get_retrain_queue() -> RetrainQueue:
    """
    재학습 큐 싱글톤 인스턴스

    Returns:
        RetrainQueue 인스턴스
    """
    global _queue_instance

    if _queue_instance is None:
        logger.info("🔧 재학습 큐 초기화 중...")
        _queue_instance = RetrainQueue()
        logger.info("✅ 재학습 큐 준비 완료")

    return _queue_instance
