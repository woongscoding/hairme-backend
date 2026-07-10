"""Upload content validation utilities

파일 확장자 문자열만 검사하던 기존 방식을 보완하여,
실제 파일 내용(매직 바이트 + Pillow 디코딩)과 크기를 서버에서 검증합니다.
"""

import io
import re

from fastapi import HTTPException
from PIL import Image

from core.exceptions import InvalidFileFormatException

# 업로드 허용 최대 크기 (Content-Length 헤더와 무관하게 실제 바이트 기준)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB

# 허용 최대 픽셀 수 (decompression bomb 완화)
MAX_IMAGE_PIXELS = 25_000_000  # 25MP (예: 5000x5000)

# 허용 확장자 (파일명 기준 1차 필터)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Gemini 프롬프트에 삽입되는 사용자 입력 길이 제한
MAX_HAIRSTYLE_NAME_LENGTH = 50
MAX_ADDITIONAL_INSTRUCTIONS_LENGTH = 200

# 제어 문자(개행 포함) 제거용 패턴
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def detect_image_format(image_data: bytes) -> str:
    """
    매직 바이트로 실제 이미지 형식을 판별

    Args:
        image_data: 업로드된 파일의 바이너리

    Returns:
        "jpeg", "png", "webp" 중 하나

    Raises:
        InvalidFileFormatException: 지원 형식이 아닌 경우
    """
    if len(image_data) >= 3 and image_data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(image_data) >= 8 and image_data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if (
        len(image_data) >= 12
        and image_data[:4] == b"RIFF"
        and image_data[8:12] == b"WEBP"
    ):
        return "webp"
    raise InvalidFileFormatException()


def validate_image_upload(
    image_data: bytes,
    max_size: int = MAX_UPLOAD_SIZE,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> str:
    """
    업로드된 이미지의 크기와 실제 내용을 검증

    Content-Length 헤더는 chunked 전송으로 우회할 수 있으므로
    반드시 body를 읽은 뒤 실제 크기로 검증해야 합니다.
    매직 바이트 확인 후 Pillow로 실제 디코딩 가능 여부와
    픽셀 수(decompression bomb 완화)까지 검증합니다.

    Args:
        image_data: 업로드된 파일의 바이너리
        max_size: 허용 최대 크기 (bytes)
        max_pixels: 허용 최대 픽셀 수 (width * height)

    Returns:
        판별된 이미지 형식 ("jpeg", "png", "webp")

    Raises:
        HTTPException(413): 크기 초과
        HTTPException(400): 해상도 초과
        InvalidFileFormatException: 이미지가 아니거나 손상된 파일
    """
    if len(image_data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=f"파일 크기가 {max_size // (1024 * 1024)}MB를 초과합니다.",
        )
    if not image_data:
        raise InvalidFileFormatException()

    image_format = detect_image_format(image_data)

    # Pillow로 실제 디코딩 검증 (헤더만 위조된 손상 파일 차단)
    try:
        with Image.open(io.BytesIO(image_data)) as img:
            width, height = img.size
            if width * height > max_pixels:
                raise HTTPException(
                    status_code=400,
                    detail="이미지 해상도가 너무 큽니다. 더 작은 이미지를 사용해주세요.",
                )
            img.verify()
    except HTTPException:
        raise
    except Exception:
        raise InvalidFileFormatException()

    return image_format


def validate_file_extension(filename: str) -> None:
    """파일명 확장자 1차 검증 (내용 검증은 validate_image_upload에서 수행)"""
    if not filename:
        raise HTTPException(status_code=400, detail="파일명이 없습니다")
    file_ext = filename.lower().rsplit(".", 1)[-1]
    if file_ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileFormatException()


def sanitize_prompt_text(
    value: str, max_length: int, field_name: str = "입력값"
) -> str:
    """
    Gemini 프롬프트에 삽입되는 사용자 입력 정제

    제어 문자와 개행 제거, 길이 제한으로 프롬프트 구조 조작 위험을
    완화합니다. 자연어 지시문 자체를 막지는 못하므로, 프롬프트 측에서도
    사용자 입력을 데이터로만 취급하도록 우선순위를 고정해야 합니다.

    Args:
        value: 사용자 입력 문자열
        max_length: 허용 최대 길이
        field_name: 오류 메시지에 표시할 필드명

    Returns:
        정제된 문자열

    Raises:
        HTTPException(400): 길이 초과 또는 정제 후 빈 문자열
    """
    sanitized = _CONTROL_CHARS.sub(" ", value)
    sanitized = " ".join(sanitized.split()).strip()

    if len(sanitized) > max_length:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name}은(는) {max_length}자를 초과할 수 없습니다.",
        )
    return sanitized
