"""SQLAlchemy database models for HairMe Backend"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class AnalysisHistory(Base):
    """Analysis history table - v20.2.0 (MediaPipe transition complete)"""
    __tablename__ = "analysis_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(100), default="anonymous")
    image_hash = Column(String(64), index=True)
    face_shape = Column(String(50))
    personal_color = Column(String(50))
    recommendations = Column(JSON)
    processing_time = Column(Float)
    detection_method = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    # OpenCV measurement data - horizontal ratios
    opencv_face_ratio = Column(Float)
    opencv_forehead_ratio = Column(Float)
    opencv_cheekbone_ratio = Column(Float)
    opencv_jaw_ratio = Column(Float)
    opencv_prediction = Column(String(50))
    opencv_confidence = Column(Float)
    opencv_gemini_agreement = Column(Boolean)

    # OpenCV measurement data - vertical ratios (v20.1.6)
    opencv_upper_face_ratio = Column(Float)
    opencv_middle_face_ratio = Column(Float)
    opencv_lower_face_ratio = Column(Float)

    # v20: Recommended styles storage
    recommended_styles = Column(JSON)

    # v20: Feedback columns (stored as String to prevent type mismatch)
    style_1_feedback = Column(String(10), nullable=True)
    style_2_feedback = Column(String(10), nullable=True)
    style_3_feedback = Column(String(10), nullable=True)

    # v20: Naver click tracking
    style_1_naver_clicked = Column(Boolean, default=False)
    style_2_naver_clicked = Column(Boolean, default=False)
    style_3_naver_clicked = Column(Boolean, default=False)

    # v20: Feedback submission timestamp
    feedback_at = Column(DateTime, nullable=True)
