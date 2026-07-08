"""카카오 로그인 검증 서비스

앱(Android)이 카카오 SDK로 발급받은 액세스 토큰을 서버로 보내면,
카카오 사용자 정보 API(/v2/user/me)로 토큰을 검증하고 프로필을 가져온다.
서버에 카카오 앱 키가 필요 없는 방식 (사용자 액세스 토큰만으로 검증).
"""

from typing import Dict, Any, Optional

import httpx
from fastapi import HTTPException, status

from core.logging import logger

KAKAO_USER_ME_URL = "https://kapi.kakao.com/v2/user/me"


class KakaoAuthService:
    """카카오 액세스 토큰 검증 및 프로필 조회"""

    async def verify_access_token(self, kakao_access_token: str) -> Dict[str, Any]:
        """
        카카오 액세스 토큰 검증

        Returns:
            {
                "kakao_id": str,       # 카카오 회원번호 (문자열로 변환)
                "nickname": str,
                "email": Optional[str],
            }

        Raises:
            HTTPException(401): 토큰이 유효하지 않음
            HTTPException(503): 카카오 API 장애
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    KAKAO_USER_ME_URL,
                    headers={"Authorization": f"Bearer {kakao_access_token}"},
                )
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
        }


# Singleton
_kakao_auth_service: Optional[KakaoAuthService] = None


def get_kakao_auth_service() -> KakaoAuthService:
    global _kakao_auth_service
    if _kakao_auth_service is None:
        _kakao_auth_service = KakaoAuthService()
    return _kakao_auth_service
