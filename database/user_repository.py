"""사용자 리포지토리 (DynamoDB hairme-users 테이블)

테이블 구조:
- Partition Key: user_id (String, UUID)
- GSI: kakao_id-index (Partition Key: kakao_id)
- Attributes: nickname, email, credits(N), training_consent(BOOL),
  created_at, last_login_at, status
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_plain(item: Dict[str, Any]) -> Dict[str, Any]:
    """DynamoDB Decimal 등을 JSON 직렬화 가능한 타입으로 변환"""
    plain = dict(item)
    if "credits" in plain:
        plain["credits"] = int(plain["credits"])
    return plain


class UserRepository:
    """사용자 CRUD (DynamoDB)"""

    KAKAO_GSI_NAME = "kakao_id-index"

    def __init__(self):
        self._table = None

    @property
    def table(self):
        """Lazy load the DynamoDB table resource"""
        if self._table is None:
            if not BOTO3_AVAILABLE:
                raise RuntimeError("boto3 is not installed")

            aws_region = os.getenv("AWS_REGION", settings.AWS_REGION)
            table_name = os.getenv(
                "DYNAMODB_USERS_TABLE_NAME",
                settings.DYNAMODB_USERS_TABLE_NAME,
            )
            config = Config(
                connect_timeout=5, read_timeout=10, retries={"max_attempts": 3}
            )
            resource = boto3.resource("dynamodb", region_name=aws_region, config=config)
            self._table = resource.Table(table_name)
        return self._table

    def get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """user_id로 사용자 조회"""
        try:
            response = self.table.get_item(Key={"user_id": user_id})
        except ClientError as e:
            logger.error(f"사용자 조회 실패: {e.response['Error']['Message']}")
            raise

        item = response.get("Item")
        return _to_plain(item) if item else None

    def get_by_kakao_id(self, kakao_id: str) -> Optional[Dict[str, Any]]:
        """카카오 회원번호로 사용자 조회 (GSI)"""
        try:
            response = self.table.query(
                IndexName=self.KAKAO_GSI_NAME,
                KeyConditionExpression="kakao_id = :kid",
                ExpressionAttributeValues={":kid": kakao_id},
                Limit=1,
            )
        except ClientError as e:
            logger.error(f"카카오 ID 조회 실패: {e.response['Error']['Message']}")
            raise

        items = response.get("Items", [])
        return _to_plain(items[0]) if items else None

    def create(
        self,
        kakao_id: str,
        nickname: str,
        email: Optional[str] = None,
        initial_credits: int = 0,
    ) -> Dict[str, Any]:
        """신규 사용자 생성 (가입 보너스 크레딧 포함)"""
        now = _now_iso()
        item: Dict[str, Any] = {
            "user_id": uuid.uuid4().hex,
            "kakao_id": kakao_id,
            "nickname": nickname,
            "credits": initial_credits,
            "training_consent": False,  # AI 학습 활용 동의는 별도 opt-in
            "status": "active",
            "created_at": now,
            "last_login_at": now,
        }
        if email:
            item["email"] = email

        try:
            self.table.put_item(
                Item=item,
                ConditionExpression="attribute_not_exists(user_id)",
            )
        except ClientError as e:
            logger.error(f"사용자 생성 실패: {e.response['Error']['Message']}")
            raise

        logger.info(f"✅ 신규 회원 가입: user_id={item['user_id']}")
        return _to_plain(item)

    def update_last_login(self, user_id: str) -> None:
        """마지막 로그인 시각 갱신"""
        try:
            self.table.update_item(
                Key={"user_id": user_id},
                UpdateExpression="SET last_login_at = :now",
                ExpressionAttributeValues={":now": _now_iso()},
            )
        except ClientError as e:
            # 로그인 시각 갱신 실패는 로그인 자체를 막을 이유가 아님
            logger.warning(f"last_login_at 갱신 실패: {e.response['Error']['Message']}")

    def set_training_consent(self, user_id: str, consent: bool) -> None:
        """원본 사진 AI 학습 활용 동의 설정 (선택 동의)"""
        try:
            self.table.update_item(
                Key={"user_id": user_id},
                UpdateExpression="SET training_consent = :c, consent_updated_at = :now",
                ConditionExpression="attribute_exists(user_id)",
                ExpressionAttributeValues={":c": consent, ":now": _now_iso()},
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise ValueError("존재하지 않는 사용자입니다")
            logger.error(f"동의 설정 실패: {e.response['Error']['Message']}")
            raise

        logger.info(f"사용자 학습 동의 변경: user_id={user_id}, consent={consent}")


# Singleton
_user_repository: Optional[UserRepository] = None


def get_user_repository() -> UserRepository:
    global _user_repository
    if _user_repository is None:
        _user_repository = UserRepository()
    return _user_repository
