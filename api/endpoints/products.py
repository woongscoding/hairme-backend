"""제휴 제품 추천/클릭 엔드포인트 (쿠팡파트너스)

- GET /products/recommendations: 스타일 기반 제품 추천 (affiliate_url 미포함)
- POST /products/click: 클릭 로그 기록 + 제휴 링크 발급 (로그인 필수)

disclosure(대가성 문구)는 공정위 표시광고 의무사항이다. 클라이언트는 추천 제품이
노출되는 영역에 이 문구를 반드시 함께 표시해야 한다.

affiliate_url을 추천 응답에 넣지 않고 클릭 엔드포인트로만 발급하는 이유:
모든 클릭이 로그를 거치게 해서 "헤어 프로필 × 제품 클릭" 데이터가 누락되지 않게.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.jwt_auth import get_current_user_id
from core.logging import logger
from services.affiliate_click_service import get_affiliate_click_service
from services.product_recommendation_service import (
    get_product_recommendation_service,
)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class ClickRequest(BaseModel):
    product_id: str = Field(..., max_length=64, description="카탈로그 product_id")
    style: Optional[str] = Field(
        None, max_length=50, description="클릭 시점의 헤어스타일 이름 (선택)"
    )
    source: str = Field(
        "browse",
        pattern="^(synthesis_result|analysis_result|browse)$",
        description="클릭이 발생한 화면",
    )
    hair_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="유저 헤어 프로필 스냅샷 (얼굴형/퍼스널컬러 등, 앱이 보유한 분석 결과)",
    )


@router.get("/products/recommendations")
@limiter.limit("60/minute")
async def get_product_recommendations(
    request: Request,
    style: Optional[str] = Query(
        None, max_length=50, description="헤어스타일 이름 (예: C컬펌)"
    ),
    gender: Optional[str] = Query(
        None, pattern="^(male|female)$", description="성별 (컷 계열 제품 선택에 사용)"
    ),
):
    """
    스타일 기반 제휴 제품 추천 (비로그인 조회 가능)

    affiliate_url은 포함되지 않는다 - POST /products/click으로 발급받을 것.
    모르는 스타일이어도 404 대신 기본 케어 제품을 반환한다.
    """
    service = get_product_recommendation_service()
    try:
        products = service.get_recommendations(style, gender=gender, limit=3)
        disclosure = service.disclosure
    except Exception:
        logger.error("❌ 제품 추천 실패", exc_info=True)
        raise HTTPException(status_code=500, detail="제품 추천에 실패했습니다.")

    return {"style": style, "products": products, "disclosure": disclosure}


@router.post("/products/click")
@limiter.limit("30/minute")
async def click_product(
    request: Request,
    body: ClickRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    제휴 제품 클릭: 클릭 로그 기록 후 제휴 링크 반환 (로그인 필수)

    로그 기록이 실패해도 affiliate_url은 반환한다 (수익 기회 > 로그 1건).
    """
    product = get_product_recommendation_service().get_product(body.product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="존재하지 않는 제품입니다.")

    try:
        # 동기 boto3 호출이라 스레드풀로 - 이벤트 루프 차단 방지
        await run_in_threadpool(
            get_affiliate_click_service().log_click,
            user_id,
            product,
            body.style,
            body.source,
            body.hair_profile,
        )
    except Exception as e:
        # log_click은 내부에서 예외를 삼키지만, 서비스 초기화 실패 등도 방어
        logger.error(f"❌ 클릭 로그 처리 실패 (링크는 발급): {str(e)}")

    return {
        "success": True,
        "product_id": product["product_id"],
        "affiliate_url": product.get("affiliate_url"),
        "disclosure": get_product_recommendation_service().disclosure,
    }
