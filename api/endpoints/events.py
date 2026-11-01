"""앱 행동 이벤트 수집 (암묵 피드백)

POST /api/events/style-action
    합성 결과 화면의 저장/공유 버튼. 앱은 공유 시트를 연 시점(취소 포함)에 share 를
    보낸다. 헤어컬러 합성 결과에는 호출하지 않는다.

    기록은 best effort 다. 스키마 위반(422)이 아니면 기록 성공 여부와 무관하게
    200 {"success": true} 를 돌려주고, 학습 샘플 저장은 응답 뒤(BackgroundTasks)에서 한다.
"""

from typing import Literal, Optional

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.jwt_auth import TOKEN_TYPE_ACCESS, decode_token
from core.logging import log_structured
from services.mlops.implicit_signals import parse_analysis_id, record_style_signal

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class StyleActionRequest(BaseModel):
    analysis_id: str = Field(..., max_length=128, description="분석 ID")
    style_name: str = Field(..., max_length=200, description="합성한 스타일 이름")
    hairstyle_id: Optional[int] = Field(None, description="추천 카드의 hairstyle_id")
    action: Literal["save", "share"]
    result_id: Optional[str] = Field(
        None, max_length=128, description="합성 응답의 result_id"
    )
    device_id: Optional[str] = Field(None, max_length=256, description="비로그인 식별자")


def _optional_user_id(request: Request) -> Optional[str]:
    """로그인 선택. 토큰이 깨졌거나 만료돼도 이벤트는 받는다 (401 을 내지 않는다)"""
    header = request.headers.get("authorization") or ""
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    try:
        return decode_token(token.strip(), TOKEN_TYPE_ACCESS).get("sub")
    except Exception:
        return None


@router.post("/events/style-action")
@limiter.limit("30/minute")
async def style_action(
    request: Request,
    body: StyleActionRequest,
    background_tasks: BackgroundTasks,
):
    """합성 결과 저장/공유 이벤트 -> 암묵 피드백 샘플 (best effort)"""
    user_id = _optional_user_id(request)
    # user_id / device_id 원문은 로그·학습 샘플에 남기지 않는다
    log_structured(
        "style_action",
        {
            "analysis_id": parse_analysis_id(body.analysis_id),
            "action": body.action,
            "hairstyle_id": body.hairstyle_id,
            "style_name": body.style_name,
            "result_id": body.result_id,
            "authenticated": user_id is not None,
            "has_device_id": bool(body.device_id),
        },
    )
    background_tasks.add_task(
        record_style_signal,
        body.analysis_id,
        body.action,
        hairstyle_id=body.hairstyle_id,
        style_name=body.style_name,
    )
    return {"success": True}
