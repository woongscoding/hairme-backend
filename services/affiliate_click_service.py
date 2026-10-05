"""제휴 링크 클릭 로그 서비스 (DynamoDB)

클릭 로그는 수수료 정산용이 아니라 데이터 자산이다:
"어떤 헤어 프로필의 유저가 어떤 스타일에서 어떤 제품을 클릭했는가"를
헤어 프로필 스냅샷과 함께 남긴다 (추후 자체 제품/추천 모델 강화의 근거 데이터).

테이블: hairme-affiliate-clicks (콘솔에서 생성 필요)
- Partition Key: user_id (S) - 로그인은 user_id, 비로그인은 "device#<device_id>"
- Sort Key: sk (S) - "<ISO8601>#<uuid8>" (시간순 정렬, 동일 초 충돌 방지)
- 속성: product_id, category, style, source, price_krw, hair_profile, created_at,
       authenticated (bool) - 두 세그먼트의 클릭률을 나눠 보기 위한 플래그
"""

import json
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

try:
    import boto3
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger, mask_user_id

# hair_profile 스냅샷 크기 상한 (비정상적으로 큰 payload 저장 방지)
MAX_HAIR_PROFILE_JSON_LENGTH = 4000


def _to_dynamo_safe(value: Any) -> Any:
    """DynamoDB는 float를 거부하므로 Decimal로 변환 (JSON 직렬화 왕복)"""
    return json.loads(json.dumps(value, ensure_ascii=False), parse_float=Decimal)


class AffiliateClickService:
    """제휴 제품 클릭 로그 기록 (best-effort - 실패해도 링크 발급은 막지 않음)"""

    def __init__(self):
        self._table = None

    @property
    def table(self):
        if self._table is None:
            if not BOTO3_AVAILABLE:
                raise RuntimeError("boto3 is not installed")
            aws_region = os.getenv("AWS_REGION", settings.AWS_REGION)
            config = Config(
                connect_timeout=5, read_timeout=10, retries={"max_attempts": 3}
            )
            table_name = os.getenv(
                "DYNAMODB_AFFILIATE_CLICKS_TABLE_NAME",
                settings.DYNAMODB_AFFILIATE_CLICKS_TABLE_NAME,
            )
            self._table = boto3.resource(
                "dynamodb", region_name=aws_region, config=config
            ).Table(table_name)
        return self._table

    def log_click(
        self,
        user_id: str,
        product: Dict[str, Any],
        style: Optional[str],
        source: str,
        hair_profile: Optional[Dict[str, Any]] = None,
        authenticated: bool = True,
    ) -> bool:
        """
        클릭 로그 기록. 실패 시 False 반환 (예외를 밖으로 던지지 않는다 -
        로그 1건보다 제휴 링크 발급이 우선).

        Args:
            user_id: 로그인은 user_id, 비로그인은 "device#<device_id>" (파티션 키)
            authenticated: 로그인 세그먼트인지. 비로그인 개방 후 두 세그먼트의
                클릭률을 나눠 보려면 식별자 접두사 파싱 대신 이 플래그를 쓴다.
        """
        now = datetime.now(timezone.utc).isoformat()
        item: Dict[str, Any] = {
            "user_id": user_id,
            "sk": f"{now}#{uuid.uuid4().hex[:8]}",
            "product_id": product.get("product_id"),
            "category": product.get("category"),
            "source": source,
            "created_at": now,
            "authenticated": authenticated,
        }
        if style:
            item["style"] = style
        if product.get("price_krw") is not None:
            item["price_krw"] = int(product["price_krw"])
        if hair_profile:
            profile_json = json.dumps(hair_profile, ensure_ascii=False)
            if len(profile_json) > MAX_HAIR_PROFILE_JSON_LENGTH:
                logger.warning(
                    f"⚠️ hair_profile 스냅샷이 너무 커서 제외: "
                    f"user_id={mask_user_id(user_id)}, len={len(profile_json)}"
                )
            else:
                item["hair_profile"] = _to_dynamo_safe(hair_profile)

        try:
            self.table.put_item(Item=item)
        except Exception as e:
            logger.error(
                f"❌ 제휴 클릭 로그 기록 실패: "
                f"user_id={mask_user_id(user_id)}, {str(e)}"
            )
            return False

        # 식별자는 마스킹한다. 비로그인 주체에는 device_id가 들어가고, device_id 원본을
        # 로그에 남기지 않는 것이 core/quota.py에서 정한 규약이다.
        logger.info(
            f"🛒 제휴 클릭: user_id={mask_user_id(user_id)}, "
            f"product={item['product_id']}, style={style}, source={source}, "
            f"authenticated={authenticated}"
        )
        return True


# Singleton
_affiliate_click_service: Optional[AffiliateClickService] = None


def get_affiliate_click_service() -> AffiliateClickService:
    global _affiliate_click_service
    if _affiliate_click_service is None:
        _affiliate_click_service = AffiliateClickService()
    return _affiliate_click_service
