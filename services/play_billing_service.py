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
PLAY_PRODUCTS_ACK_URL = PLAY_PRODUCTS_GET_URL + ":acknowledge"
PLAY_VOIDED_PURCHASES_URL = (
    "https://androidpublisher.googleapis.com/androidpublisher/v3"
    "/applications/{package_name}/purchases/voidedpurchases"
)

# voidedpurchases 의 type 파라미터: 0=인앱 상품(일회성), 1=구독 포함
VOIDED_TYPE_ONE_TIME = 0

# purchases.products.get 응답의 purchaseState 값
PURCHASE_STATE_PURCHASED = 0
PURCHASE_STATE_CANCELED = 1
PURCHASE_STATE_PENDING = 2

# purchases.products.get 응답의 acknowledgementState 값
ACK_STATE_NOT_ACKNOWLEDGED = 0
ACK_STATE_ACKNOWLEDGED = 1


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
            # 0=미승인(3일 내 미승인 시 Google이 자동 환불), 1=승인됨
            "acknowledgement_state": int(
                data.get("acknowledgementState", ACK_STATE_NOT_ACKNOWLEDGED)
            ),
        }

    def acknowledge_product_purchase(
        self, product_id: str, purchase_token: str
    ) -> None:
        """
        구매 서버 측 승인 (purchases.products.acknowledge)

        미승인 구매는 3일 후 Google이 자동 환불하므로, 크레딧 지급 전에
        서버가 직접 승인해 "자동 환불 + 크레딧 유지" 공격/사고를 차단한다.

        이미 승인된 구매에 대한 400 응답은 성공으로 간주한다
        (동시 요청 race 또는 앱이 먼저 승인한 경우).

        Raises:
            PlayBillingUnavailableError: Google API 장애 (클라이언트 재시도 유도)
        """
        url = PLAY_PRODUCTS_ACK_URL.format(
            package_name=quote(self.package_name, safe=""),
            product_id=quote(product_id, safe=""),
            token=quote(purchase_token, safe=""),
        )

        try:
            response = self.session.post(url, json={}, timeout=10)
        except PlayBillingError:
            raise
        except Exception:
            logger.error("❌ Play acknowledge 호출 실패 (네트워크)", exc_info=True)
            raise PlayBillingUnavailableError("play acknowledge request failed")

        if response.status_code in (200, 204):
            logger.info(f"✅ Play 구매 승인 완료: product={product_id}")
            return

        if response.status_code == 400:
            # 이미 승인된 구매를 다시 승인하면 400 - 지급 진행에 문제 없음
            logger.warning(
                f"⚠️ Play acknowledge 400 (이미 승인된 구매로 간주): "
                f"product={product_id}"
            )
            return

        logger.error(f"❌ Play acknowledge 오류: status={response.status_code}")
        raise PlayBillingUnavailableError("play acknowledge error")

    def list_voided_purchases(
        self,
        start_time_millis: int,
        page_token: Optional[str] = None,
        voided_type: int = VOIDED_TYPE_ONE_TIME,
    ) -> Dict[str, Any]:
        """
        환불/취소/차지백된 구매 목록 1페이지 조회 (purchases.voidedpurchases.list)

        페이지네이션은 호출자가 응답의 tokenPagination.nextPageToken 을
        page_token 으로 넘겨 반복한다.

        Args:
            start_time_millis: 조회 시작 시각 (epoch ms, Play 는 최대 30일 과거까지)
            page_token: 다음 페이지 토큰 (첫 페이지는 None)
            voided_type: 0=인앱 상품만, 1=구독 포함

        Returns:
            Play API 원본 JSON ({"voidedPurchases": [...], "tokenPagination": {...}})

        Raises:
            PlayBillingUnavailableError: 설정 누락 또는 Google API 장애
        """
        if not self.package_name:
            logger.error("❌ PLAY_PACKAGE_NAME이 설정되지 않음 (환불 조회 불가)")
            raise PlayBillingUnavailableError("package name not configured")

        url = PLAY_VOIDED_PURCHASES_URL.format(
            package_name=quote(self.package_name, safe=""),
        )
        params: Dict[str, Any] = {
            "startTime": str(int(start_time_millis)),
            "type": int(voided_type),
        }
        if page_token:
            params["token"] = page_token

        try:
            response = self.session.get(url, params=params, timeout=15)
        except PlayBillingError:
            raise
        except Exception:
            logger.error("❌ voidedpurchases 호출 실패 (네트워크)", exc_info=True)
            raise PlayBillingUnavailableError("voidedpurchases request failed")

        if response.status_code != 200:
            logger.error(f"❌ voidedpurchases API 오류: status={response.status_code}")
            raise PlayBillingUnavailableError(
                f"voidedpurchases api error: {response.status_code}"
            )

        return response.json() or {}


# Singleton
_play_billing_service: Optional[PlayBillingService] = None


def get_play_billing_service() -> PlayBillingService:
    global _play_billing_service
    if _play_billing_service is None:
        _play_billing_service = PlayBillingService()
    return _play_billing_service
