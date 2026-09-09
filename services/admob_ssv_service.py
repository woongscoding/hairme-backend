"""AdMob 리워드 광고 서버측 검증(SSV) 서비스

AdMob이 광고 시청 완료 시 우리 서버로 보내는 콜백의 ECDSA(P-256/SHA-256)
서명을 검증한다. 위조된 콜백으로 보상 크레딧을 챙기는 것을 방지.

- 공개키: https://www.gstatic.com/admob/reward/verifier-keys.json 에서 로드 후 캐싱
- 서명 대상: 원본 쿼리 스트링에서 "&signature=" 직전까지의 바이트
  (Google Tink RewardedAdsVerifier 레퍼런스 구현과 동일 - URL 디코딩 없이 원본 그대로)
- 반환 파라미터는 "서명된 구간"에서만 파싱한다. 서명 뒤에 붙은 값
  (예: &user_id=victim)은 서명 대상이 아니므로 신뢰할 수 없다.
"""

import base64
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple
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

# 서명 구간 뒤(= 서명 대상이 아닌 구간)에 올 수 있는 유일한 파라미터.
# AdMob은 message 뒤에 signature, key_id 순으로만 덧붙인다.
UNSIGNED_ALLOWED_KEYS = frozenset({"signature", "key_id"})

# 리워드 콜백 신선도 (AdMob timestamp는 밀리초 단위)
REWARD_MAX_AGE_SECONDS = 600  # 10분보다 오래된 콜백은 재사용 시도로 간주
REWARD_MAX_FUTURE_SECONDS = 300  # 시계 오차 허용치 (5분)


def parse_ad_unit_allowlist(raw: Optional[str]) -> Set[str]:
    """쉼표로 구분된 ad_unit 허용 목록 문자열을 집합으로 파싱"""
    if not raw:
        return set()
    return {part.strip() for part in raw.split(",") if part.strip()}


def is_reward_timestamp_fresh(
    raw_timestamp: str, now_seconds: Optional[float] = None
) -> bool:
    """AdMob timestamp(밀리초)가 허용 구간 안에 있는지 검사

    파싱 불가 / 과거 10분 초과 / 미래 5분 초과면 False.
    """
    try:
        timestamp_seconds = int(str(raw_timestamp).strip()) / 1000.0
    except (TypeError, ValueError):
        return False

    now = time.time() if now_seconds is None else now_seconds
    age = now - timestamp_seconds
    return -REWARD_MAX_FUTURE_SECONDS <= age <= REWARD_MAX_AGE_SECONDS


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

    @staticmethod
    def _parse_pairs(chunk: bytes) -> List[Tuple[str, str]]:
        """쿼리 조각을 (key, value) 리스트로 파싱 (중복 키 보존)"""
        return parse_qsl(
            chunk.decode("utf-8", errors="replace"), keep_blank_values=True
        )

    @staticmethod
    def _has_duplicate_keys(keys: Iterable[str]) -> bool:
        seen: Set[str] = set()
        for key in keys:
            if key in seen:
                return True
            seen.add(key)
        return False

    def verify_callback(self, raw_query: bytes) -> Dict[str, str]:
        """
        SSV 콜백 쿼리 스트링 검증 후 "서명된 구간"의 파라미터만 반환

        Args:
            raw_query: URL 디코딩되지 않은 원본 쿼리 스트링 바이트
                       (request.scope["query_string"])

        Returns:
            서명 대상 구간에서 파싱한 쿼리 파라미터
            (user_id, transaction_id, reward_amount 등)

        Raises:
            InvalidSSVError: 서명 불일치 / signature·key_id 누락 /
                             서명 구간 중복 키 / 서명 뒤 파라미터 덧붙이기
            SSVUnavailableError: 공개키 조회 불가
        """
        # 서명 대상: "&signature=" 앞까지의 원본 바이트
        sig_index = raw_query.find(SIGNATURE_PARAM)
        if sig_index <= 0:
            logger.warning("⚠️ SSV 콜백 쿼리 형식 오류 (signature 위치)")
            raise InvalidSSVError("malformed query string")
        message = raw_query[:sig_index]
        # 선행 "&"를 제외한 서명 이후 구간 ("signature=...&key_id=...")
        unsigned_tail = raw_query[sig_index + 1 :]

        # 1) 서명 구간: 중복 키가 있으면 어떤 값이 서명됐는지 모호해지므로 거부
        signed_pairs = self._parse_pairs(message)
        if self._has_duplicate_keys(key for key, _ in signed_pairs):
            logger.warning("⚠️ SSV 콜백 서명 구간에 중복 파라미터")
            raise InvalidSSVError("duplicate parameters in signed portion")
        if any(key in UNSIGNED_ALLOWED_KEYS for key, _ in signed_pairs):
            logger.warning("⚠️ SSV 콜백 서명 구간에 signature/key_id 포함")
            raise InvalidSSVError("signature parameters inside signed portion")
        params = dict(signed_pairs)

        # 2) 서명 이후 구간: signature, key_id 외에는 무엇도 올 수 없다.
        #    (허용하면 서명된 user_id/transaction_id를 뒤에서 덮어쓸 수 있음)
        tail_pairs = self._parse_pairs(unsigned_tail)
        if self._has_duplicate_keys(key for key, _ in tail_pairs):
            logger.warning("⚠️ SSV 콜백 서명 이후 구간에 중복 파라미터")
            raise InvalidSSVError("duplicate parameters after signature")
        extra_keys = [key for key, _ in tail_pairs if key not in UNSIGNED_ALLOWED_KEYS]
        if extra_keys:
            logger.warning(f"⚠️ SSV 콜백 서명 이후 미서명 파라미터 덧붙임: {extra_keys}")
            raise InvalidSSVError("unsigned parameters after signature")

        tail_params = dict(tail_pairs)
        signature = tail_params.get("signature")
        key_id = tail_params.get("key_id")
        if not signature or not key_id:
            logger.warning("⚠️ SSV 콜백에 signature/key_id 누락")
            raise InvalidSSVError("missing signature or key_id")

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
