"""크레딧 조회/구매 엔드포인트"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import logger
from core.jwt_auth import get_current_user_id
from services.credit_service import get_credit_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class PurchaseRequest(BaseModel):
    product_id: str = Field(..., description="스토어 상품 ID (예: credits_10)")
    purchase_token: str = Field(..., description="Google Play 구매 토큰")


@router.get("/credits")
async def get_credits(user_id: str = Depends(get_current_user_id)):
    """크레딧 잔액 + 최근 증감 내역 조회"""
    try:
        service = get_credit_service()
        balance = service.get_balance(user_id)
        history = service.get_history(user_id, limit=20)
    except ValueError:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception as e:
        logger.error(f"❌ 크레딧 조회 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="크레딧 조회에 실패했습니다.")

    return {"balance": balance, "history": history}


@router.post("/credits/purchase")
@limiter.limit("10/minute")
async def purchase_credits(
    request: Request,
    body: PurchaseRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    크레딧 구매 (Google Play 인앱결제 영수증 검증)

    TODO: Google Play Developer API 연동 필요
    1. purchases.products.get으로 purchase_token 검증
    2. 검증 성공 시 credit_service.grant(user_id, amount, "purchase", ref_id=purchase_token)
    3. purchase_token 중복 사용 방지 (원장에서 ref_id 조회)

    Play Console 서비스 계정 설정 전까지는 501을 반환한다.
    """
    raise HTTPException(
        status_code=501,
        detail="크레딧 구매는 아직 준비 중입니다.",
    )
