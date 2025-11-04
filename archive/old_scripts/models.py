# models.py
from sqlalchemy import Column, Integer, String, DECIMAL, ForeignKey
from sqlalchemy.orm import relationship
from database import Base # 1단계에서 만든 Base 임포트

# 2단계에서 만든 SQL 테이블과 1:1로 매핑되는 Python 클래스들

class FaceShape(Base):
    __tablename__ = "face_shapes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True)

class SkinTone(Base):
    __tablename__ = "skin_tones"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, unique=True)

class Hairstyle(Base):
    __tablename__ = "hairstyles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)

class Recommendation(Base):
    __tablename__ = "recommendations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    face_shape_id = Column(Integer, ForeignKey("face_shapes.id"), nullable=False)
    skin_tone_id = Column(Integer, ForeignKey("skin_tones.id"), nullable=False)
    hairstyle_id = Column(Integer, ForeignKey("hairstyles.id"), nullable=False)
    score = Column(DECIMAL(3, 2), default=0.90)
    reason = Column(String(255))

    # Python 코드에서 .hairstyle, .face_shape 등으로 바로 접근할 수 있게 설정
    hairstyle = relationship("Hairstyle")
    face_shape = relationship("FaceShape")
    skin_tone = relationship("SkinTone")