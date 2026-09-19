"""FastAPI dependencies and Pydantic models"""

from enum import Enum
from typing import Optional, Union
from pydantic import BaseModel, Field


# ========== Enums ==========
class FeedbackType(str, Enum):
    """Feedback type enumeration - supports multiple formats for backward compatibility"""

    GOOD = "good"
    BAD = "bad"
    LIKE = "like"  # Backward compatibility
    DISLIKE = "dislike"  # Backward compatibility


class DislikeReason(str, Enum):
    """싫어요 사유 (선택 입력) - 품질 문제의 종류를 나누기 위한 최소 집합"""

    FACE_CHANGED = "face_changed"  # 얼굴이 바뀌어 보임
    HAIR_UNNATURAL = "hair_unnatural"  # 머리가 부자연스러움
    NOT_MY_TASTE = "not_my_taste"  # 결과는 멀쩡하지만 취향이 아님


# ========== Pydantic Models ==========
class FeedbackRequest(BaseModel):
    """Feedback submission request"""

    analysis_id: Union[int, str] = Field(
        ..., description="분석 결과 ID (DynamoDB UUID 문자열)"
    )
    style_index: int = Field(
        ..., ge=1, le=5, description="스타일 인덱스 (1-5, 4-5는 트렌드 스타일)"
    )
    feedback: FeedbackType = Field(..., description="좋아요 또는 싫어요")
    naver_clicked: bool = Field(
        default=False, description="네이버 이미지 검색 클릭 여부"
    )
    # 선택 필드 - 구버전 앱은 보내지 않으며, 없으면 기존과 동일하게 처리된다
    dislike_reason: Optional[DislikeReason] = Field(
        default=None,
        description="싫어요 사유 (선택): face_changed | hair_unnatural | not_my_taste",
    )


class FeedbackResponse(BaseModel):
    """Feedback submission response"""

    success: bool
    message: str
    analysis_id: Union[int, str]  # DynamoDB UUID 문자열 (레거시 int 응답 호환 유지)
    style_index: int
