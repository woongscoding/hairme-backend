"""FastAPI dependencies and Pydantic models"""

from enum import Enum
from pydantic import BaseModel, Field


# ========== Enums ==========
class FeedbackType(str, Enum):
    """Feedback type enumeration - supports multiple formats for backward compatibility"""
    GOOD = "good"
    BAD = "bad"
    LIKE = "like"  # Backward compatibility
    DISLIKE = "dislike"  # Backward compatibility


# ========== Pydantic Models ==========
class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    analysis_id: int = Field(..., description="분석 결과 ID")
    style_index: int = Field(..., ge=1, le=3, description="스타일 인덱스 (1, 2, 3)")
    feedback: FeedbackType = Field(..., description="좋아요 또는 싫어요")
    naver_clicked: bool = Field(default=False, description="네이버 이미지 검색 클릭 여부")


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    success: bool
    message: str
    analysis_id: int
    style_index: int
