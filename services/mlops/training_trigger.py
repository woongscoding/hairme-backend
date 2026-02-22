"""
재학습 트리거 서비스

EventBridge 또는 데이터 임계값 기반으로 재학습을 트리거합니다.

트리거 조건:
1. 시간 기반: EventBridge 스케줄 (예: 매주 일요일 03:00 UTC)
2. 데이터 기반: pending 피드백이 RETRAIN_THRESHOLD 이상

비용:
    - EventBridge: 무료 (월 14,000,000 이벤트까지)
    - Lambda 호출: 무료 (월 100만 요청까지)

Author: HairMe ML Team
Date: 2025-12-02
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration
RETRAIN_THRESHOLD = int(os.getenv("MLOPS_RETRAIN_THRESHOLD", "100"))
TRAINER_LAMBDA_NAME = os.getenv("MLOPS_TRAINER_LAMBDA", "hairme-model-trainer")
SNS_TOPIC_ARN = os.getenv("MLOPS_SNS_TOPIC_ARN", "")  # 알림용 (선택)

try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class TrainingTrigger:
    """
    재학습 트리거 관리

    EventBridge 규칙 생성 및 Lambda 호출을 담당합니다.
    """

    def __init__(self, region: str = "ap-northeast-2"):
        """
        초기화

        Args:
            region: AWS 리전
        """
        self.region = region
        self.lambda_client = None
        self.events_client = None
        self.sns_client = None
        self.enabled = False

        self._init_clients()

    def _init_clients(self):
        """AWS 클라이언트 초기화"""
        if not BOTO3_AVAILABLE:
            logger.warning("⚠️ boto3 not installed - Training trigger 비활성화")
            return

        mlops_enabled = os.getenv("MLOPS_ENABLED", "false").lower() == "true"
        if not mlops_enabled:
            logger.info("ℹ️ MLOps 비활성화 (MLOPS_ENABLED=false)")
            return

        try:
            self.lambda_client = boto3.client("lambda", region_name=self.region)
            self.events_client = boto3.client("events", region_name=self.region)

            if SNS_TOPIC_ARN:
                self.sns_client = boto3.client("sns", region_name=self.region)

            self.enabled = True
            logger.info("✅ Training trigger 초기화 완료")

        except Exception as e:
            logger.error(f"❌ AWS 클라이언트 초기화 실패: {e}")

    def check_and_trigger(self, pending_count: int) -> Dict[str, Any]:
        """
        데이터 기반 재학습 트리거 확인 및 실행

        Args:
            pending_count: 대기 중인 피드백 수

        Returns:
            {
                "triggered": bool,
                "reason": str,
                "invocation_id": str (optional)
            }
        """
        if not self.enabled:
            return {"triggered": False, "reason": "MLOps disabled"}

        if pending_count < RETRAIN_THRESHOLD:
            return {
                "triggered": False,
                "reason": f"Threshold not met ({pending_count}/{RETRAIN_THRESHOLD})",
            }

        # 트리거 실행
        return self.trigger_training(
            trigger_type="data_threshold",
            metadata={"pending_count": pending_count, "threshold": RETRAIN_THRESHOLD},
        )

    def trigger_training(
        self, trigger_type: str = "manual", metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        재학습 Lambda 호출

        Args:
            trigger_type: 트리거 유형 ("manual", "data_threshold", "scheduled")
            metadata: 추가 메타데이터

        Returns:
            {
                "triggered": bool,
                "reason": str,
                "invocation_id": str
            }
        """
        if not self.enabled:
            return {"triggered": False, "reason": "MLOps disabled"}

        try:
            payload = {
                "trigger_type": trigger_type,
                "triggered_at": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }

            response = self.lambda_client.invoke(
                FunctionName=TRAINER_LAMBDA_NAME,
                InvocationType="Event",  # 비동기 호출
                Payload=json.dumps(payload),
            )

            invocation_id = response.get("ResponseMetadata", {}).get(
                "RequestId", "unknown"
            )

            logger.info(
                f"✅ 재학습 트리거 성공: type={trigger_type}, "
                f"invocation_id={invocation_id}"
            )

            # SNS 알림 (설정된 경우)
            if self.sns_client and SNS_TOPIC_ARN:
                self._send_notification(
                    subject=f"[HairMe MLOps] 재학습 시작 ({trigger_type})",
                    message=json.dumps(payload, indent=2, ensure_ascii=False),
                )

            return {
                "triggered": True,
                "reason": f"Training triggered ({trigger_type})",
                "invocation_id": invocation_id,
            }

        except ClientError as e:
            error_msg = e.response["Error"]["Message"]
            logger.error(f"❌ 재학습 트리거 실패: {error_msg}")
            return {
                "triggered": False,
                "reason": f"Lambda invocation failed: {error_msg}",
            }

        except Exception as e:
            logger.error(f"❌ 재학습 트리거 실패: {e}")
            return {"triggered": False, "reason": f"Error: {str(e)}"}

    def _send_notification(self, subject: str, message: str):
        """SNS 알림 발송"""
        try:
            self.sns_client.publish(
                TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message
            )
            logger.info(f"✅ SNS 알림 발송: {subject}")
        except Exception as e:
            logger.warning(f"⚠️ SNS 알림 실패: {e}")

    def setup_scheduled_rule(
        self,
        rule_name: str = "hairme-weekly-retrain",
        schedule: str = "cron(0 3 ? * SUN *)",  # 매주 일요일 03:00 UTC
        enabled: bool = True,
    ) -> Dict[str, Any]:
        """
        EventBridge 스케줄 규칙 생성

        Args:
            rule_name: 규칙 이름
            schedule: cron 표현식 (UTC 기준)
            enabled: 활성화 여부

        Returns:
            규칙 ARN 또는 에러 정보
        """
        if not self.enabled:
            return {"success": False, "reason": "MLOps disabled"}

        try:
            # 규칙 생성/업데이트
            response = self.events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule,
                State="ENABLED" if enabled else "DISABLED",
                Description="HairMe ML 모델 주간 재학습 스케줄",
            )

            rule_arn = response["RuleArn"]

            # Lambda 타겟 설정
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        "Id": "trainer-lambda",
                        "Arn": f"arn:aws:lambda:{self.region}:{self._get_account_id()}:function:{TRAINER_LAMBDA_NAME}",
                        "Input": json.dumps(
                            {"trigger_type": "scheduled", "schedule": schedule}
                        ),
                    }
                ],
            )

            logger.info(f"✅ EventBridge 규칙 설정 완료: {rule_name}")

            return {"success": True, "rule_arn": rule_arn, "schedule": schedule}

        except Exception as e:
            logger.error(f"❌ EventBridge 규칙 설정 실패: {e}")
            return {"success": False, "reason": str(e)}

    def _get_account_id(self) -> str:
        """AWS 계정 ID 조회"""
        try:
            sts = boto3.client("sts", region_name=self.region)
            return sts.get_caller_identity()["Account"]
        except Exception:
            return "000000000000"

    def get_rule_status(
        self, rule_name: str = "hairme-weekly-retrain"
    ) -> Dict[str, Any]:
        """
        EventBridge 규칙 상태 조회

        Args:
            rule_name: 규칙 이름

        Returns:
            규칙 상태 정보
        """
        if not self.enabled:
            return {"enabled": False}

        try:
            response = self.events_client.describe_rule(Name=rule_name)

            return {
                "name": response["Name"],
                "state": response["State"],
                "schedule": response.get("ScheduleExpression", ""),
                "arn": response["Arn"],
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                return {"name": rule_name, "state": "NOT_FOUND"}
            raise

        except Exception as e:
            logger.error(f"❌ 규칙 상태 조회 실패: {e}")
            return {"error": str(e)}


# ========== 싱글톤 인스턴스 ==========
_trigger_instance = None


def get_training_trigger() -> TrainingTrigger:
    """
    Training trigger 싱글톤 인스턴스

    Returns:
        TrainingTrigger 인스턴스
    """
    global _trigger_instance

    if _trigger_instance is None:
        logger.info("🔧 Training trigger 초기화 중...")
        _trigger_instance = TrainingTrigger()

    return _trigger_instance
