# models/face_analyzer.py
import cv2
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

# 로거 초기화
logger = logging.getLogger(__name__)


@dataclass
class FaceFeatures:
    """거리 무관 얼굴 비율 특징"""
    # 기본 비율
    face_ratio: float  # 높이/너비 (가장 중요!)

    # 얼굴 부위별 비율 (얼굴 너비 대비)
    forehead_ratio: float  # 이마 너비 / 얼굴 너비
    cheekbone_ratio: float  # 광대 너비 / 얼굴 너비
    jaw_ratio: float  # 턱 너비 / 얼굴 너비

    # 수직 비율
    upper_face_ratio: float  # 이마 높이 / 얼굴 높이
    middle_face_ratio: float  # 중안부 높이 / 얼굴 높이
    lower_face_ratio: float  # 하안부 높이 / 얼굴 높이

    # 예측 힌트
    face_shape_hint: str
    confidence: float  # 예측 신뢰도 (0~1)

    def to_dict(self) -> dict:
        """dict로 변환 (로깅 및 DB 저장용)"""
        return {
            "face_ratio": self.face_ratio,
            "forehead_ratio": self.forehead_ratio,
            "cheekbone_ratio": self.cheekbone_ratio,
            "jaw_ratio": self.jaw_ratio,
            "upper_face_ratio": self.upper_face_ratio,
            "middle_face_ratio": self.middle_face_ratio,
            "lower_face_ratio": self.lower_face_ratio,
            "face_shape_hint": self.face_shape_hint,
            "confidence": self.confidence
        }


def extract_face_features(image_data: bytes) -> Optional[FaceFeatures]:
    """
    OpenCV로 얼굴 비율 특징 추출 (거리 무관)

    얼굴형 판별 기준:
    - 계란형: 0.95 < ratio < 1.25, 균형잡힌 3등분
    - 둥근형: ratio < 1.0, 턱이 둥글고 넓음
    - 각진형: 광대 > 이마/턱, 각진 턱선
    - 긴형: ratio > 1.3, 세로로 길쭉함
    """
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            logger.error("이미지 디코딩 실패")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 얼굴 검출
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            logger.warning("OpenCV로 얼굴을 검출하지 못함")
            return None

        # 가장 큰 얼굴 선택
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y + h, x:x + w]

        # ========== 1. 기본 얼굴 비율 (가장 중요!) ==========
        face_ratio = h / w

        # ========== 2. 수평 비율 (3등분으로 나눔) ==========
        # 이마 영역 (상단 1/3)
        upper_third = face_roi[0:int(h * 0.33), :]
        # 중안부 영역 (중간 1/3)
        middle_third = face_roi[int(h * 0.33):int(h * 0.67), :]
        # 하안부 영역 (하단 1/3)
        lower_third = face_roi[int(h * 0.67):, :]

        # 각 영역의 수평 너비 측정 (엣지 검출로 윤곽 파악)
        def measure_width(roi):
            """ROI의 평균 수평 너비 측정"""
            edges = cv2.Canny(roi, 50, 150)
            # 각 행에서 엣지가 있는 최좌/최우 픽셀 거리
            widths = []
            for row in edges:
                points = np.where(row > 0)[0]
                if len(points) > 1:
                    widths.append(points[-1] - points[0])
            return np.mean(widths) if widths else w * 0.5

        forehead_width = measure_width(upper_third)
        cheekbone_width = measure_width(middle_third)
        jaw_width = measure_width(lower_third)

        # ========== 3. 비율 계산 (얼굴 너비 w로 정규화) ==========
        forehead_ratio = forehead_width / w
        cheekbone_ratio = cheekbone_width / w
        jaw_ratio = jaw_width / w

        # 수직 비율 (각 영역 높이는 동일하므로 0.33)
        upper_face_ratio = 0.33
        middle_face_ratio = 0.33
        lower_face_ratio = 0.33

        # ========== 4. 얼굴형 판별 로직 ==========
        confidence = 0.0

        if face_ratio > 1.35:
            # 긴형: 세로가 매우 길다
            face_shape_hint = "긴형"
            confidence = min((face_ratio - 1.35) * 2, 0.9)

        elif face_ratio < 0.95:
            # 둥근형: 가로가 넓다
            face_shape_hint = "둥근형"
            confidence = min((0.95 - face_ratio) * 2, 0.9)

        elif cheekbone_ratio > forehead_ratio * 1.1 and cheekbone_ratio > jaw_ratio * 1.15:
            # 각진형: 광대가 이마/턱보다 확실히 넓다
            face_shape_hint = "각진형"
            confidence = min(
                (cheekbone_ratio - max(forehead_ratio, jaw_ratio)) * 3,
                0.85
            )

        elif 1.0 <= face_ratio <= 1.3 and abs(forehead_ratio - jaw_ratio) < 0.1:
            # 계란형: 비율 균형, 이마/턱 비슷
            face_shape_hint = "계란형"
            balance_score = 1 - abs(forehead_ratio - jaw_ratio) * 5
            confidence = min(balance_score * 0.8, 0.8)

        else:
            # 애매한 경우
            face_shape_hint = "계란형"  # 기본값
            confidence = 0.5

        logger.info(f"OpenCV 특징 추출 성공: {face_shape_hint} (신뢰도: {confidence:.0%})")

        return FaceFeatures(
            face_ratio=round(face_ratio, 3),
            forehead_ratio=round(forehead_ratio, 3),
            cheekbone_ratio=round(cheekbone_ratio, 3),
            jaw_ratio=round(jaw_ratio, 3),
            upper_face_ratio=upper_face_ratio,
            middle_face_ratio=middle_face_ratio,
            lower_face_ratio=lower_face_ratio,
            face_shape_hint=face_shape_hint,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        logger.error(f"얼굴 특징 추출 실패: {str(e)}")
        return None


def create_enhanced_prompt(features: FaceFeatures) -> str:
    """
    OpenCV 측정값을 반영한 개선된 Gemini 프롬프트 생성

    Args:
        features: OpenCV로 측정한 얼굴 특징

    Returns:
        OpenCV 힌트가 포함된 프롬프트
    """
    return f"""다음 얼굴 사진을 분석하고 JSON으로 응답해주세요.

🔍 **참고용 측정 데이터** (OpenCV 자동 분석):
- 얼굴 비율(높이/너비): {features.face_ratio:.2f}
- 이마 너비 비율: {features.forehead_ratio:.2f}
- 광대 너비 비율: {features.cheekbone_ratio:.2f}
- 턱 너비 비율: {features.jaw_ratio:.2f}
- OpenCV 예측: {features.face_shape_hint} (신뢰도: {features.confidence:.0%})

위 수치는 참고만 하고, 당신의 시각적 판단을 우선하세요.

**분석 항목:**
1. 얼굴형: 계란형/둥근형/각진형/긴형 중 1개
2. 퍼스널컬러: 봄웜/가을웜/여름쿨/겨울쿨 중 1개
3. 헤어스타일 추천 3개 (각 이름 15자, 이유 30자 이내)

**JSON 형식:**
{{
  "analysis": {{
    "face_shape": "계란형",
    "personal_color": "봄웜",
    "features": "이목구비 특징 설명"
  }},
  "recommendations": [
    {{"style_name": "스타일명", "reason": "추천 이유"}}
  ]
}}"""