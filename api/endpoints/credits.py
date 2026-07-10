"""크레딧 조회/구매 엔드포인트"""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from config.settings import settings
from core.logging import logger
from core.jwt_auth import get_current_user_id
from services.credit_service import get_credit_service
from services.play_billing_service import (
    InvalidPurchaseError,
    PlayBillingUnavailableError,
    get_play_billing_service,
)

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

    1. purchases.products.get으로 purchase_token 검증
    2. purchase_token 클레임 마커(조건부 put)로 중복 지급 방지
    3. credit_service.grant(reason="purchase") - 지급 실패 시 마커 회수
    """
    amount = settings.CREDIT_PRODUCTS.get(body.product_id)
    if amount is None:
        raise HTTPException(status_code=400, detail="알 수 없는 상품입니다.")

    # 영수증 검증 (동기 requests 호출이라 스레드풀로 - 이벤트 루프 차단 방지)
    play_service = get_play_billing_service()
    try:
        receipt = await run_in_threadpool(
            play_service.verify_product_purchase, body.product_id, body.purchase_token
        )
    except InvalidPurchaseError:
        logger.warning(
            f"⚠️ 구매 영수증 검증 실패: user_id={user_id}, product={body.product_id}"
        )
        raise HTTPException(status_code=400, detail="구매 영수증 검증에 실패했습니다.")
    except PlayBillingUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="결제 검증 서비스를 일시적으로 사용할 수 없습니다.",
        )

    # 같은 purchase_token 재사용 시 중복 지급 방지 (멱등성)
    credit_service = get_credit_service()
    ref_key = f"purchase#{body.purchase_token}"
    if not credit_service.try_claim_ref(
        ref_key,
        user_id,
        detail={"product_id": body.product_id, "order_id": receipt.get("order_id")},
    ):
        raise HTTPException(status_code=409, detail="이미 처리된 구매입니다.")

    try:
        balance = credit_service.grant(
            user_id, amount, reason="purchase", ref_id=body.purchase_token
        )
    except ValueError:
        credit_service.release_ref(ref_key)
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    except Exception:
        # 지급 실패 시 클레임 회수 - 클라이언트가 재시도할 수 있게
        credit_service.release_ref(ref_key)
        logger.error(f"❌ 구매 크레딧 지급 실패: user_id={user_id}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="크레딧 지급 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.",
        )

    logger.info(
        f"💳 크레딧 구매 완료: user_id={user_id}, product={body.product_id}, "
        f"+{amount}, 잔액={balance}"
    )
    return {
        "success": True,
        "product_id": body.product_id,
        "granted": amount,
        "balance": balance,
        "order_id": receipt.get("order_id"),
    }
