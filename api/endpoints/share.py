"""Pro 계정(미용실용 앱) 결과 공유 페이지 - 손님이 QR 로 여는 짧은 주소

GET /s/{token}
- token 은 합성 때 서버가 만든 128비트 난수(22자)다. 주소를 아는 사람만 열 수 있고 추측할 수 없다.
- 결과를 만든 뒤 settings.PRO_RESULT_RETENTION_DAYS(7일) 동안만 열린다.
- 이미지는 열 때마다 새로 만든 짧은 presigned URL 로 보여 준다.
  (합성 응답의 result_url 은 Lambda 임시 자격증명 때문에 24시간보다 일찍 끊길 수 있다)
- 로그인이 필요 없다. 손님 폰의 브라우저가 연다.
- 로그에는 토큰을 남기지 않는다 (토큰 = 열람 권한).
"""

from datetime import datetime, timedelta, timezone
from html import escape

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from core.logging import log_structured, logger
from services.photo_storage_service import ShareLinkExpired, get_photo_storage_service

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

SHARE_PATH_PREFIX = "/s/"

KST = timezone(timedelta(hours=9))

# 공유 페이지는 검색에 노출되거나 캐시에 남으면 안 된다.
# Referrer-Policy 는 보안 헤더 미들웨어가 strict-origin-when-cross-origin 으로 덮어쓴다
# (S3 이미지 요청에는 주소 경로 = 토큰이 실리지 않는다).
_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
}


def share_url(request: Request, token: str, base_url: str = "") -> str:
    """공유 토큰의 전체 주소. base_url 이 비면 요청이 들어온 서버 주소를 쓴다."""
    base = (base_url or str(request.base_url)).rstrip("/")
    return f"{base}{SHARE_PATH_PREFIX}{token}"


def _page(status_code: int, title: str, body: str) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
body{{margin:0;padding:16px;font-family:sans-serif;background:#faf7f5;color:#2b211b;text-align:center}}
img{{max-width:100%;height:auto;border-radius:12px}}
p{{line-height:1.5}}
.note{{color:#6b5b50;font-size:14px}}
</style></head><body>{body}</body></html>"""
    return HTMLResponse(content=html, status_code=status_code, headers=_PAGE_HEADERS)


def _message_page(status_code: int, message: str) -> HTMLResponse:
    return _page(
        status_code,
        "Hair Consultant",
        f"<h2>Hair Consultant</h2><p>{escape(message)}</p>",
    )


@router.get(SHARE_PATH_PREFIX + "{token}", response_class=HTMLResponse)
@limiter.limit("30/minute")
async def open_shared_result(request: Request, token: str):
    """손님에게 보낸 합성 결과 보기 (로그인 불필요)"""
    try:
        found = await run_in_threadpool(
            get_photo_storage_service().resolve_share_token, token
        )
    except ShareLinkExpired:
        log_structured("pro_share_opened", {"result": "expired"})
        return _message_page(410, "보관 기간이 지나 사진이 삭제되었어요.")
    except Exception as e:
        logger.error(f"❌ 공유 주소 조회 실패: {type(e).__name__}")
        log_structured("pro_share_opened", {"result": "error"})
        return _message_page(503, "잠시 후 다시 열어 주세요.")

    if found is None:
        log_structured("pro_share_opened", {"result": "not_found"})
        return _message_page(404, "주소가 맞지 않거나 사진이 삭제되었어요.")

    log_structured("pro_share_opened", {"result": "ok"})
    until = found["expires_at"].astimezone(KST)
    until_text = f"{until.month}월 {until.day}일 {until.hour:02d}:{until.minute:02d}"
    return _page(
        200,
        "헤어스타일 상담 결과",
        f"<h2>헤어스타일 상담 결과</h2>"
        f'<img src="{escape(found["image_url"])}" alt="합성 결과">'
        f"<p>사진을 길게 눌러 저장할 수 있어요.</p>"
        f'<p class="note">AI 가 만든 참고용 이미지예요. 실제 시술 결과와 다를 수 있어요.<br>'
        f"이 사진은 {escape(until_text)}까지 볼 수 있고, 그 뒤 자동으로 삭제돼요.</p>",
    )


def iso_expires_at(created_at: datetime, days: int) -> str:
    """공유 주소 만료 시각 (UTC ISO 8601, 앱 표시용)"""
    return (created_at + timedelta(days=days)).astimezone(timezone.utc).isoformat()
