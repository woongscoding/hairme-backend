"""AdMob 리워드 광고 서버측 검증(SSV) 서비스

AdMob이 광고 시청 완료 시 우리 서버로 보내는 콜백의 ECDSA(P-256/SHA-256)
서명을 검증한다. 위조된 콜백으로 보상 크레딧을 챙기는 것을 방지.

- 공개키: https://www.gstatic.com/admob/reward/verifier-keys.json 에서 로드 후 캐싱
- 서명 대상: 원본 쿼리 스트링에서 "&signature=" 직전까지의 바이트
  (Google Tink RewardedAdsVerifier 레퍼런스 구현과 동일 - URL 디코딩 없이 원본 그대로)
"""

import base64
import time
from typing import Dict, Optional
from urllib.parse import parse_qsl

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_pem_public_key

from core.logging import logger

VERIFIER_KEYS_URL = "https://www.gstatic.com/admob/reward/verifier-keys.json"
KEYS_CACHE_TTL_SECONDS = 86400  # 공개키 캐시 24시간 (Google이 주기적으로 회전)

SIGNATURE_PARAM = b"&signature="


class AdMobSSVError(Exception):
    """SSV 검증 오류 (base)"""


class InvalidSSVError(AdMobSSVError):
    """서명 검증 실패 / 필수 파라미터 누락 → 400"""


class SSVUnavailableError(AdMobSSVError):
    """공개키 조회 불가 (Google 서버 장애) → 503"""


class AdMobSSVService:
    """AdMob SSV 콜백 서명 검증"""

    def __init__(self):
        self._keys: Dict[str, str] = {}  # key_id → PEM 공개키
        self._keys_fetched_at: float = 0.0

    def _fetch_keys(self) -> Dict[str, str]:
        """Google 검증 키 목록 다운로드"""
        try:
            response = httpx.get(VERIFIER_KEYS_URL, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        except Exception:
            logger.error("❌ AdMob 검증 키 조회 실패", exc_info=True)
            raise SSVUnavailableError("verifier keys fetch failed")

        keys = {
            str(entry["keyId"]): entry["pem"]
            for entry in data.get("keys", [])
            if entry.get("keyId") is not None and entry.get("pem")
        }
        if not keys:
            logger.error("❌ AdMob 검증 키 응답이 비어 있음")
            raise SSVUnavailableError("no verifier keys in response")
        return keys

    def _get_key_pem(self, key_id: str) -> Optional[str]:
        """key_id에 해당하는 공개키 PEM (캐시 만료/키 회전 시 재조회)"""
        now = time.monotonic()
        cache_expired = (
            not self._keys or now - self._keys_fetched_at > KEYS_CACHE_TTL_SECONDS
        )

        if cache_expired or key_id not in self._keys:
            self._keys = self._fetch_keys()
            self._keys_fetched_at = now

        return self._keys.get(key_id)

    def verify_callback(self, raw_query: bytes) -> Dict[str, str]:
        """
        SSV 콜백 쿼리 스트링 검증 후 파라미터 반환

        Args:
            raw_query: URL 디코딩되지 않은 원본 쿼리 스트링 바이트
                       (request.scope["query_string"])

        Returns:
            파싱된 쿼리 파라미터 (user_id, transaction_id, reward_amount 등)

        Raises:
            InvalidSSVError: 서명 불일치 / signature·key_id 누락
            SSVUnavailableError: 공개키 조회 불가
        """
        params = dict(parse_qsl(raw_query.decode("utf-8", errors="replace")))
        signature = params.get("signature")
        key_id = params.get("key_id")
        if not signature or not key_id:
            logger.warning("⚠️ SSV 콜백에 signature/key_id 누락")
            raise InvalidSSVError("missing signature or key_id")

        # 서명 대상: "&signature=" 앞까지의 원본 바이트
        sig_index = raw_query.find(SIGNATURE_PARAM)
        if sig_index <= 0:
            logger.warning("⚠️ SSV 콜백 쿼리 형식 오류 (signature 위치)")
            raise InvalidSSVError("malformed query string")
        message = raw_query[:sig_index]

        pem = self._get_key_pem(key_id)
        if pem is None:
            logger.warning(f"⚠️ 알 수 없는 SSV key_id: {key_id}")
            raise InvalidSSVError("unknown key_id")

        try:
            # 서명은 URL-safe base64 (패딩 생략 가능)
            signature_bytes = base64.urlsafe_b64decode(
                signature + "=" * (-len(signature) % 4)
            )
            public_key = load_pem_public_key(pem.encode("utf-8"))
            public_key.verify(signature_bytes, message, ec.ECDSA(hashes.SHA256()))
        except InvalidSignature:
            logger.warning("⚠️ SSV 서명 검증 실패 (위조 가능성)")
            raise InvalidSSVError("signature verification failed")
        except InvalidSSVError:
            raise
        except Exception:
            logger.warning("⚠️ SSV 서명 형식 오류", exc_info=True)
            raise InvalidSSVError("malformed signature")

        return params


# Singleton
_admob_ssv_service: Optional[AdMobSSVService] = None


def get_admob_ssv_service() -> AdMobSSVService:
    global _admob_ssv_service
    if _admob_ssv_service is None:
        _admob_ssv_service = AdMobSSVService()
    return _admob_ssv_service
