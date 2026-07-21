"""헤어스타일 예시 이미지 서빙 엔드포인트

Lambda 이미지에 번들된 WebP(data_source/style_images/webp/)를 서빙한다.
이미지는 배포 시점에 고정되므로 1년 캐시를 허용한다
(스타일 이미지가 바뀌면 파일명이 달라져 캐시 무효화 문제 없음).
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()

_WEBP_DIR = (
    Path(__file__).parent.parent.parent / "data_source" / "style_images" / "webp"
)

_CACHE_HEADERS = {"Cache-Control": "public, max-age=31536000, immutable"}


@router.get("/styles/images/{filename}")
async def get_style_image(filename: str):
    """스타일 예시 이미지 WebP 반환"""
    # 경로 조작 방지: 구분자/상위 이동 문자를 포함하면 거부
    if (
        "/" in filename
        or "\\" in filename
        or ".." in filename
        or not filename.endswith(".webp")
    ):
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

    path = _WEBP_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

    return FileResponse(path, media_type="image/webp", headers=_CACHE_HEADERS)
