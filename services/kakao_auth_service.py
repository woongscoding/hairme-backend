"""카카오 로그인 검증 서비스

앱(Android)이 카카오 SDK로 발급받은 액세스 토큰을 서버로 보내면,
카카오 사용자 정보 API(/v2/user/me)로 토큰을 검증하고 프로필을 가져온다.
서버에 카카오 앱 키가 필요 없는 방식 (사용자 액세스 토큰만으로 검증).
"""

from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException, status

from config.settings import settings
from core.account_type import ACCOUNT_CUSTOMER, ACCOUNT_PRO
from core.logging import logger

KAKAO_USER_ME_URL = "https://kapi.kakao.com/v2/user/me"
KAKAO_TOKEN_INFO_URL = "https://kapi.kakao.com/v1/user/access_token_info"


class KakaoAuthService:
    """카카오 액세스 토큰 검증 및 프로필 조회"""

    @staticmethod
    def _pro_app_id(customer_app_id: str) -> str:
        """Pro 카카오 앱 ID (미설정이거나 손님 앱 ID 와 같으면 빈 문자열)"""
        pro_app_id = str(settings.KAKAO_PRO_APP_ID or "").strip()
        if pro_app_id and pro_app_id == customer_app_id:
            logger.error(
                "❌ KAKAO_PRO_APP_ID 가 KAKAO_APP_ID 와 같습니다 - Pro 로그인을 끕니다"
            )
            return ""
        return pro_app_id

    async def _verify_token_app_id(
        self, client: httpx.AsyncClient, kakao_access_token: str
    ) -> str:
        """
        토큰이 우리 카카오 앱에서 발급된 것인지 검증하고 계정 유형을 정한다

        공격자가 자신이 통제하는 다른 카카오 앱의 토큰을 제출하는
        confused-deputy 시나리오를 access_token_info의 app_id로 걸러낸다.
        KAKAO_APP_ID 미설정 시 손님 앱 검증을 생략한다 (기존 배포 호환).

        계정 유형은 토큰을 발급한 카카오 앱으로만 정한다 (클라이언트 입력을 믿지 않는다).
        - app_id == KAKAO_PRO_APP_ID → Pro
        - 그 밖 → 손님 (KAKAO_APP_ID 가 설정돼 있으면 일치해야 한다)

        Returns:
            ACCOUNT_CUSTOMER 또는 ACCOUNT_PRO
        """
        customer_app_id = str(settings.KAKAO_APP_ID or "").strip()
        pro_app_id = self._pro_app_id(customer_app_id)

        if not customer_app_id and not pro_app_id:
            logger.warning(
                "⚠️ KAKAO_APP_ID 미설정 - 카카오 토큰 발급 앱 검증 생략 "
                "(프로덕션에서는 설정 권장)"
            )
            return ACCOUNT_CUSTOMER

        response = await client.get(
            KAKAO_TOKEN_INFO_URL,
            headers={"Authorization": f"Bearer {kakao_access_token}"},
        )

        if response.status_code == 401:
            logger.warning("⚠️ 유효하지 않은 카카오 토큰 (access_token_info)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="카카오 인증에 실패했습니다. 다시 로그인해주세요.",
            )
        if response.status_code != 200:
            logger.error(
                f"❌ 카카오 access_token_info 오류: status={response.status_code}"
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="카카오 로그인 서비스에 일시적 오류가 발생했습니다.",
            )

        token_app_id = str(response.json().get("app_id", ""))
        if pro_app_id and token_app_id == pro_app_id:
            return ACCOUNT_PRO
        if not customer_app_id:
            # 손님 앱 ID 미설정 - 기존처럼 검증 없이 손님으로 처리
            return ACCOUNT_CUSTOMER
        if token_app_id != customer_app_id:
            logger.warning(
                f"⚠️ 다른 카카오 앱에서 발급된 토큰으로 로그인 시도 차단: "
                f"token_app_id={token_app_id}"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="카카오 인증에 실패했습니다. 다시 로그인해주세요.",
            )
        return ACCOUNT_CUSTOMER

    async def verify_access_token(self, kakao_access_token: str) -> Dict[str, Any]:
        """
        카카오 액세스 토큰 검증

        1. access_token_info로 토큰 발급 앱(app_id)이 우리 앱인지 확인 + 계정 유형 결정
        2. /v2/user/me로 프로필 조회

        Returns:
            {
                "kakao_id": str,       # 카카오 회원번호 (문자열로 변환)
                "nickname": str,
                "email": Optional[str],
                "account_type": "customer" | "pro",
            }

        Raises:
            HTTPException(401): 토큰이 유효하지 않거나 다른 앱에서 발급됨
            HTTPException(503): 카카오 API 장애
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                account_type = await self._verify_token_app_id(
                    client, kakao_access_token
                )
                response = await client.get(
                    KAKAO_USER_ME_URL,
                    headers={"Authorization": f"Bearer {kakao_access_token}"},
                )
        except HTTPException:
            raise
        except httpx.HTTPError:
            logger.error("❌ 카카오 API 호출 실패 (네트워크)", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="카카오 로그인 서비스에 일시적 오류가 발생했습니다.",
            )

        if response.status_code == 401:
            logger.warning("⚠️ 유효하지 않은 카카오 토큰으로 로그인 시도")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="카카오 인증에 실패했습니다. 다시 로그인해주세요.",
            )

        if response.status_code != 200:
            logger.error(f"❌ 카카오 API 오류: status={response.status_code}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="카카오 로그인 서비스에 일시적 오류가 발생했습니다.",
            )

        data = response.json()

        kakao_id = data.get("id")
        if kakao_id is None:
            logger.error("❌ 카카오 응답에 회원번호(id)가 없음")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="카카오 인증에 실패했습니다. 다시 로그인해주세요.",
            )

        kakao_account = data.get("kakao_account") or {}
        profile = kakao_account.get("profile") or {}

        return {
            "kakao_id": str(kakao_id),
            "nickname": profile.get("nickname") or "회원",
            "email": kakao_account.get("email"),
            "account_type": account_type,
        }


# Singleton
_kakao_auth_service: Optional[KakaoAuthService] = None


def get_kakao_auth_service() -> KakaoAuthService:
    global _kakao_auth_service
    if _kakao_auth_service is None:
        _kakao_auth_service = KakaoAuthService()
    return _kakao_auth_service
