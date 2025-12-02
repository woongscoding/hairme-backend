"""
관리자 대시보드 라우터

피드백 통계, MLOps 상태, A/B 테스트 관리 API를 제공합니다.
신버전: S3 + DynamoDB 기반 MLOps 시스템 + A/B 테스트

Author: HairMe ML Team
Date: 2025-12-02
Version: 3.0.0
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from services.circuit_breaker import get_circuit_breaker_status, reset_circuit_breakers
from core.auth import verify_admin_api_key
from api.dependencies import ABTestStartRequest, ABTestPromoteRequest
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


# ========== A/B 테스트 관리 API ==========

@router.get("/admin/abtest/status")
async def get_abtest_status(api_key: str = Depends(verify_admin_api_key)) -> Dict[str, Any]:
    """
    현재 A/B 테스트 상태 조회

    Returns:
        - enabled: A/B 테스트 활성화 여부
        - experiment_id: 현재 실험 ID
        - champion_version: Champion 모델 버전
        - challenger_version: Challenger 모델 버전
        - challenger_traffic_percent: Challenger 트래픽 비율
        - started_at: 실험 시작 시간
    """
    try:
        from services.mlops.ab_test import get_ab_router

        router = get_ab_router()
        config = router.config

        return {
            "success": True,
            "enabled": config.enabled,
            "experiment_id": config.experiment_id,
            "champion_version": config.champion_model_version,
            "challenger_version": config.challenger_model_version,
            "challenger_traffic_percent": config.challenger_traffic_percent,
            "started_at": config.started_at,
            "is_active": router.is_abtest_active()
        }

    except ImportError:
        return {
            "success": False,
            "enabled": False,
            "message": "A/B 테스트 모듈이 로드되지 않았습니다"
        }
    except Exception as e:
        logger.error(f"❌ A/B 테스트 상태 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"A/B 테스트 상태 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/abtest/metrics/{experiment_id}")
async def get_abtest_metrics(
    experiment_id: str,
    api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    특정 실험의 A/B 테스트 지표 조회

    Args:
        experiment_id: 실험 ID (예: "exp_2025_12_02")

    Returns:
        - champion_metrics: Champion 모델 지표
            - sample_count: 총 피드백 수
            - positive_feedback_rate: 긍정 피드백 비율
            - score_discrimination: 점수 구분력
        - challenger_metrics: Challenger 모델 지표
        - conclusion: 승자 판단 결과
        - recommendation: 권장 조치
    """
    try:
        from services.mlops.ab_evaluator import get_ab_evaluator

        evaluator = get_ab_evaluator()
        metrics = evaluator.get_metrics_by_variant(experiment_id)

        if not metrics:
            return {
                "success": False,
                "message": f"실험 '{experiment_id}'에 대한 데이터가 없습니다"
            }

        # 승자 판단
        result = evaluator.is_challenger_better(metrics)

        return {
            "success": True,
            "experiment_id": experiment_id,
            **result
        }

    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="A/B 테스트 평가기 모듈이 로드되지 않았습니다"
        )
    except Exception as e:
        logger.error(f"❌ A/B 테스트 지표 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"A/B 테스트 지표 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.get("/admin/abtest/summary/{experiment_id}")
async def get_abtest_summary(
    experiment_id: str,
    api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    실험 요약 정보 조회

    Args:
        experiment_id: 실험 ID

    Returns:
        - total_samples: 총 샘플 수
        - champion_samples: Champion 샘플 수
        - challenger_samples: Challenger 샘플 수
        - current_winner: 현재 승자
    """
    try:
        from services.mlops.ab_evaluator import get_ab_evaluator

        evaluator = get_ab_evaluator()
        summary = evaluator.get_experiment_summary(experiment_id)

        return {
            "success": True,
            **summary
        }

    except Exception as e:
        logger.error(f"❌ A/B 테스트 요약 조회 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"A/B 테스트 요약 조회 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/admin/abtest/start")
async def start_abtest(
    request: ABTestStartRequest,
    api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    새 A/B 테스트 시작

    Note: 이 API는 런타임에만 설정을 변경합니다.
    영구 설정을 위해서는 환경변수를 수정해야 합니다.

    Args:
        request: ABTestStartRequest
            - experiment_id: 실험 ID
            - challenger_model_version: Challenger 모델 버전
            - challenger_traffic_percent: Challenger 트래픽 비율

    Returns:
        - success: 성공 여부
        - message: 결과 메시지
        - config: 적용된 설정
    """
    try:
        from services.mlops.ab_test import get_ab_router, ABTestConfig, refresh_ab_router

        # 새 설정으로 라우터 업데이트
        new_config = ABTestConfig(
            experiment_id=request.experiment_id,
            champion_model_version=os.getenv('ABTEST_CHAMPION_VERSION', 'v6'),
            challenger_model_version=request.challenger_model_version,
            challenger_traffic_percent=request.challenger_traffic_percent,
            enabled=True,
            started_at=datetime.now(timezone.utc).isoformat()
        )

        router = get_ab_router()
        router.update_config(new_config)

        logger.warning(
            f"⚠️ [ADMIN] A/B 테스트 시작: experiment={request.experiment_id}, "
            f"challenger={request.challenger_model_version}, traffic={request.challenger_traffic_percent}%"
        )

        return {
            "success": True,
            "message": f"A/B 테스트가 시작되었습니다 (experiment: {request.experiment_id})",
            "config": new_config.to_dict(),
            "warning": "이 설정은 런타임에만 적용됩니다. 서버 재시작 시 환경변수 설정이 필요합니다."
        }

    except Exception as e:
        logger.error(f"❌ A/B 테스트 시작 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"A/B 테스트 시작 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/admin/abtest/stop")
async def stop_abtest(api_key: str = Depends(verify_admin_api_key)) -> Dict[str, Any]:
    """
    현재 A/B 테스트 중지

    Challenger 트래픽을 0%로 설정하여 모든 요청이 Champion으로 가도록 합니다.

    Returns:
        - success: 성공 여부
        - message: 결과 메시지
    """
    try:
        from services.mlops.ab_test import get_ab_router, ABTestConfig

        router = get_ab_router()
        old_experiment = router.config.experiment_id

        # 현재 설정을 비활성화
        new_config = ABTestConfig(
            experiment_id=router.config.experiment_id,
            champion_model_version=router.config.champion_model_version,
            challenger_model_version=router.config.challenger_model_version,
            challenger_traffic_percent=0,
            enabled=False,
            started_at=router.config.started_at
        )

        router.update_config(new_config)

        logger.warning(f"⚠️ [ADMIN] A/B 테스트 중지: experiment={old_experiment}")

        return {
            "success": True,
            "message": f"A/B 테스트가 중지되었습니다 (experiment: {old_experiment})",
            "config": new_config.to_dict()
        }

    except Exception as e:
        logger.error(f"❌ A/B 테스트 중지 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"A/B 테스트 중지 중 오류가 발생했습니다: {str(e)}"
        )


@router.post("/admin/abtest/promote/{experiment_id}")
async def promote_challenger(
    experiment_id: str,
    api_key: str = Depends(verify_admin_api_key)
) -> Dict[str, Any]:
    """
    Challenger를 Champion으로 승격

    현재 Challenger 모델을 새로운 Champion으로 설정합니다.
    실제 모델 파일 교체는 별도로 수행해야 합니다.

    Args:
        experiment_id: 승격할 실험 ID

    Returns:
        - success: 성공 여부
        - message: 결과 메시지
        - new_champion_version: 새 Champion 버전
    """
    try:
        from services.mlops.ab_test import get_ab_router, ABTestConfig
        from services.mlops.ab_evaluator import get_ab_evaluator

        router = get_ab_router()

        # 현재 실험 확인
        if router.config.experiment_id != experiment_id:
            raise HTTPException(
                status_code=400,
                detail=f"현재 실험 ID({router.config.experiment_id})와 일치하지 않습니다"
            )

        # 지표 확인
        evaluator = get_ab_evaluator()
        metrics = evaluator.get_metrics_by_variant(experiment_id)
        result = evaluator.is_challenger_better(metrics)

        if result.get('conclusion') != 'challenger_wins':
            logger.warning(
                f"⚠️ Challenger가 승자가 아닌데 승격 시도: "
                f"conclusion={result.get('conclusion')}"
            )

        # Challenger를 Champion으로 승격
        new_champion_version = router.config.challenger_model_version

        new_config = ABTestConfig(
            experiment_id="",  # 실험 종료
            champion_model_version=new_champion_version,
            challenger_model_version="",
            challenger_traffic_percent=0,
            enabled=False,
            started_at=None
        )

        router.update_config(new_config)

        logger.warning(
            f"⚠️ [ADMIN] Challenger 승격 완료: "
            f"experiment={experiment_id}, new_champion={new_champion_version}"
        )

        return {
            "success": True,
            "message": f"Challenger가 Champion으로 승격되었습니다",
            "experiment_id": experiment_id,
            "new_champion_version": new_champion_version,
            "evaluation_result": result,
            "next_steps": [
                "1. S3에서 challenger/model.pt를 current/model.pt로 복사",
                "2. Lambda/ECS 환경변수 업데이트 (ABTEST_CHAMPION_VERSION)",
                "3. 서비스 재배포"
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Challenger 승격 실패: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Challenger 승격 중 오류가 발생했습니다: {str(e)}"
        )
