"""FastAPI dependencies and Pydantic models"""

from enum import Enum
from typing import Union
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
    analysis_id: Union[int, str] = Field(..., description="분석 결과 ID (int for MySQL, UUID string for DynamoDB)")
    style_index: int = Field(..., ge=1, le=3, description="스타일 인덱스 (1, 2, 3)")
    feedback: FeedbackType = Field(..., description="좋아요 또는 싫어요")
    naver_clicked: bool = Field(default=False, description="네이버 이미지 검색 클릭 여부")


class FeedbackResponse(BaseModel):
    """Feedback submission response"""
    success: bool
    message: str
    analysis_id: Union[int, str]  # Support both MySQL (int) and DynamoDB (str UUID)
    style_index: int


# ========== A/B 테스트 관련 Models ==========
class ABTestStartRequest(BaseModel):
    """A/B 테스트 시작 요청"""
    experiment_id: str = Field(..., description="실험 ID (예: exp_2025_12_02)")
    challenger_model_version: str = Field(..., description="Challenger 모델 버전")
    challenger_traffic_percent: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Challenger 트래픽 비율 (0-100)"
    )


class ABTestPromoteRequest(BaseModel):
    """A/B 테스트 승격 요청"""
    experiment_id: str = Field(..., description="승격할 실험 ID")
