"""사용자 리포지토리 (DynamoDB hairme-users 테이블)

테이블 구조:
- Partition Key: user_id (String, UUID)
- GSI: kakao_id-index (Partition Key: kakao_id)
- Attributes: nickname, email, credits(N), training_consent(BOOL),
  created_at, last_login_at, status

같은 테이블에 kakao_id 유일성 마커 아이템도 저장한다.
- user_id = "kakao#<kakao_id>", ref_user_id = 실제 user_id
- 마커에는 kakao_id 속성이 없으므로 kakao_id-index GSI에 색인되지 않는다
  (= get_by_kakao_id가 마커를 반환할 일이 없음)
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

# kakao_id 유일성 마커 아이템의 user_id 접두사
KAKAO_MARKER_PREFIX = "kakao#"


class UserAlreadyExistsError(Exception):
    """같은 kakao_id로 이미 가입된 사용자가 있음 (동시 가입 경합)"""

    def __init__(self, kakao_id: str):
        self.kakao_id = kakao_id
        super().__init__(f"이미 가입된 카카오 계정입니다: {kakao_id}")


def kakao_marker_id(kakao_id: str) -> str:
    """kakao_id 유일성 마커 아이템의 파티션 키"""
    return f"{KAKAO_MARKER_PREFIX}{kakao_id}"


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

        # kakao_id 유일성 마커 + 실제 사용자 아이템을 한 트랜잭션으로 기록.
        # GSI 조회는 최종 일관성이라 "조회 후 생성"만으로는 동시 요청에서
        # 같은 kakao_id로 계정이 여러 개 생기고 가입 보너스도 중복 지급된다.
        marker = {
            "user_id": kakao_marker_id(kakao_id),
            "ref_user_id": item["user_id"],
            "created_at": now,
        }

        try:
            table_name = self.table.name
            # 주의: DynamoDB *리소스*의 내부 클라이언트(table.meta.client)는 파이썬 값을
            # 자동으로 AttributeValue로 변환한다. 여기서 TypeSerializer로 먼저 직렬화하면
            # {"S": ...}가 다시 Map(M)으로 감싸져 "Type mismatch for key user_id
            # expected: S actual: M" ValidationError가 난다 (2026-09-09 프로덕션 가입 장애).
            # 따라서 평문 파이썬 값을 그대로 넘긴다.
            self.table.meta.client.transact_write_items(
                TransactItems=[
                    {
                        "Put": {
                            "TableName": table_name,
                            "Item": marker,
                            "ConditionExpression": "attribute_not_exists(user_id)",
                        }
                    },
                    {
                        "Put": {
                            "TableName": table_name,
                            "Item": item,
                            "ConditionExpression": "attribute_not_exists(user_id)",
                        }
                    },
                ]
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "TransactionCanceledException":
                reasons = e.response.get("CancellationReasons") or []
                marker_failed = bool(reasons) and (
                    reasons[0].get("Code") == "ConditionalCheckFailed"
                )
                if marker_failed:
                    logger.info(
                        f"동시 가입 감지 - 기존 계정으로 전환: kakao_id={kakao_id}"
                    )
                    raise UserAlreadyExistsError(kakao_id)
            logger.error(
                "사용자 생성 실패: %s | reasons=%s",
                e.response["Error"]["Message"],
                e.response.get("CancellationReasons"),
            )
            raise

        logger.info(f"✅ 신규 회원 가입: user_id={item['user_id']}")
        return _to_plain(item)

    def get_by_kakao_marker(self, kakao_id: str) -> Optional[Dict[str, Any]]:
        """유일성 마커를 강한 일관성으로 읽어 실제 사용자 조회

        GSI(get_by_kakao_id)는 최종 일관성이라 방금 생성된 계정이 보이지 않을
        수 있다. 동시 가입 경합에서 확실히 기존 계정을 찾기 위한 경로.
        """
        try:
            response = self.table.get_item(
                Key={"user_id": kakao_marker_id(kakao_id)},
                ConsistentRead=True,
            )
        except ClientError as e:
            logger.error(f"카카오 마커 조회 실패: {e.response['Error']['Message']}")
            raise

        marker = response.get("Item")
        ref_user_id = marker.get("ref_user_id") if marker else None
        if not ref_user_id:
            return None
        return self.get_by_id(ref_user_id)

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
