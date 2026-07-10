"""Google Play 인앱결제 영수증 검증 서비스

앱이 결제 완료 후 보낸 purchase_token을 Google Play Developer API
(purchases.products.get)로 검증한다.

- 인증: 서비스 계정 키(JSON)
  - Secrets Manager(hairme-play-service-account) → 환경변수/설정(PLAY_SERVICE_ACCOUNT_JSON) 폴백
- 무거운 google-api-python-client 대신 google-auth의 AuthorizedSession으로 REST 직접 호출
  (액세스 토큰 발급/갱신은 AuthorizedSession이 자동 처리)
"""

import json
import os
from typing import Any, Dict, Optional
from urllib.parse import quote

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import AuthorizedSession

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

from config.secrets import get_secret_or_env
from config.settings import settings
from core.logging import logger

PLAY_API_SCOPE = "https://www.googleapis.com/auth/androidpublisher"
PLAY_PRODUCTS_GET_URL = (
    "https://androidpublisher.googleapis.com/androidpublisher/v3"
    "/applications/{package_name}/purchases/products/{product_id}/tokens/{token}"
)

# purchases.products.get 응답의 purchaseState 값
PURCHASE_STATE_PURCHASED = 0
PURCHASE_STATE_CANCELED = 1
PURCHASE_STATE_PENDING = 2


class PlayBillingError(Exception):
    """Play 결제 검증 오류 (base)"""


class PlayBillingUnavailableError(PlayBillingError):
    """검증 서비스 사용 불가 (설정 누락 / Google API 장애) → 503"""


class InvalidPurchaseError(PlayBillingError):
    """유효하지 않은 영수증 (존재하지 않는 토큰 / 미결제 상태) → 400"""


class PlayBillingService:
    """Google Play Developer API로 인앱결제 영수증 검증"""

    def __init__(self):
        self._session = None

    @property
    def package_name(self) -> str:
        return os.getenv("PLAY_PACKAGE_NAME", settings.PLAY_PACKAGE_NAME)

    def _load_credentials(self):
        """서비스 계정 자격증명 로드 (Secrets Manager → 환경변수/설정 폴백)"""
        raw = (
            get_secret_or_env(
                secret_name="hairme-play-service-account",
                env_var_name="PLAY_SERVICE_ACCOUNT_JSON",
                region_name=settings.AWS_REGION,
                required=False,
            )
            or settings.PLAY_SERVICE_ACCOUNT_JSON
        )
        if not raw:
            logger.error("❌ Play 서비스 계정 키가 설정되지 않음 (구매 검증 불가)")
            raise PlayBillingUnavailableError("service account key not configured")

        try:
            info = json.loads(raw)
            return service_account.Credentials.from_service_account_info(
                info, scopes=[PLAY_API_SCOPE]
            )
        except (ValueError, KeyError):
            logger.error("❌ Play 서비스 계정 키 파싱 실패 (JSON 형식 확인 필요)")
            raise PlayBillingUnavailableError("invalid service account key")

    @property
    def session(self):
        """인증된 HTTP 세션 (lazy 싱글톤, 토큰 갱신 자동)"""
        if self._session is None:
            if not GOOGLE_AUTH_AVAILABLE:
                raise PlayBillingUnavailableError("google-auth is not installed")
            self._session = AuthorizedSession(self._load_credentials())
        return self._session

    def verify_product_purchase(
        self, product_id: str, purchase_token: str
    ) -> Dict[str, Any]:
        """
        인앱 상품 구매 영수증 검증

        Returns:
            {"order_id": str|None, "purchase_time_millis": str|None}

        Raises:
            InvalidPurchaseError: 토큰이 유효하지 않거나 결제 완료 상태가 아님
            PlayBillingUnavailableError: 설정 누락 또는 Google API 장애
        """
        if not self.package_name:
            logger.error("❌ PLAY_PACKAGE_NAME이 설정되지 않음 (구매 검증 불가)")
            raise PlayBillingUnavailableError("package name not configured")

        url = PLAY_PRODUCTS_GET_URL.format(
            package_name=quote(self.package_name, safe=""),
            product_id=quote(product_id, safe=""),
            token=quote(purchase_token, safe=""),
        )

        try:
            response = self.session.get(url, timeout=10)
        except PlayBillingError:
            raise
        except Exception:
            logger.error("❌ Play Developer API 호출 실패 (네트워크)", exc_info=True)
            raise PlayBillingUnavailableError("play api request failed")

        # 존재하지 않거나 다른 앱/상품의 토큰이면 Google이 400/404를 반환
        if response.status_code in (400, 404):
            logger.warning(
                f"⚠️ 유효하지 않은 구매 토큰: product={product_id}, "
                f"status={response.status_code}"
            )
            raise InvalidPurchaseError("purchase token not found")

        if response.status_code != 200:
            logger.error(f"❌ Play Developer API 오류: status={response.status_code}")
            raise PlayBillingUnavailableError("play api error")

        data = response.json()
        purchase_state = int(data.get("purchaseState", -1))
        if purchase_state != PURCHASE_STATE_PURCHASED:
            # 취소(1)/보류(2) 상태는 지급 대상이 아님
            logger.warning(
                f"⚠️ 결제 미완료 상태의 구매 토큰: product={product_id}, "
                f"purchaseState={purchase_state}"
            )
            raise InvalidPurchaseError("purchase not completed")

        logger.info(
            f"✅ Play 구매 검증 성공: product={product_id}, "
            f"orderId={data.get('orderId')}"
        )
        return {
            "order_id": data.get("orderId"),
            "purchase_time_millis": data.get("purchaseTimeMillis"),
        }


# Singleton
_play_billing_service: Optional[PlayBillingService] = None


def get_play_billing_service() -> PlayBillingService:
    global _play_billing_service
    if _play_billing_service is None:
        _play_billing_service = PlayBillingService()
    return _play_billing_service
