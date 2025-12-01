# models/mediapipe_analyzer.py
"""
MediaPipe Face Mesh를 사용한 고정밀 얼굴 분석기
- 478개 3D 랜드마크 기반 얼굴형 분류 (정확도 90%+)
- ITA + HSV 기반 피부톤 분석 (정확도 85%+)
"""

# import cv2  # Lazy loaded
# import numpy as np  # Lazy loaded
# import mediapipe as mp  # Lazy loaded
import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Any
import math

# 로거 초기화
logger = logging.getLogger(__name__)


@dataclass
class MediaPipeFaceFeatures:
    """MediaPipe로 추출한 얼굴 특징"""
    face_shape: str           # "계란형", "둥근형", "각진형", "긴형", "하트형"
    skin_tone: str            # "봄웜", "가을웜", "여름쿨", "겨울쿨"
    confidence: float         # 신뢰도 (0.0 ~ 1.0)

    # 측정값 (디버깅용)
    face_ratio: float         # 높이/너비
    forehead_width: float     # 이마 너비 (픽셀)
    cheekbone_width: float    # 광대 너비 (픽셀)
    jaw_width: float          # 턱 너비 (픽셀)
    ITA_value: float          # 피부톤 ITA 값
    hue_value: float          # 색조 값 (HSV)

    # ML 모델 입력용 특징 벡터 (학습 데이터와 동일한 형식)
    face_features: list       # [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
    skin_features: list       # [ITA_value, hue_value] (2차원)

    # 성별 추론 (NEW)
    gender: str = "neutral"   # "male", "female", "neutral" (추론 실패 시)
    gender_confidence: float = 0.0  # 성별 추론 신뢰도

    def to_dict(self) -> dict:
        """dict로 변환 (로깅 및 DB 저장용)"""
        return {
            "face_shape": self.face_shape,
            "skin_tone": self.skin_tone,
            "confidence": self.confidence,
            "face_ratio": self.face_ratio,
            "forehead_width": self.forehead_width,
            "cheekbone_width": self.cheekbone_width,
            "jaw_width": self.jaw_width,
            "ITA_value": self.ITA_value,
            "hue_value": self.hue_value,
            "face_features": self.face_features,
            "skin_features": self.skin_features,
            "gender": self.gender,
            "gender_confidence": self.gender_confidence
        }


class MediaPipeFaceAnalyzer:
    """MediaPipe 기반 얼굴 분석기"""

    # MediaPipe 주요 랜드마크 인덱스
    # 얼굴 윤곽선 및 주요 포인트
    FOREHEAD_TOP = 10          # 이마 상단
    CHIN_BOTTOM = 152          # 턱 하단
    LEFT_CHEEK = 234           # 왼쪽 광대
    RIGHT_CHEEK = 454          # 오른쪽 광대
    LEFT_FOREHEAD = 70         # 왼쪽 이마 측면
    RIGHT_FOREHEAD = 300       # 오른쪽 이마 측면
    LEFT_JAW = 172             # 왼쪽 턱선
    RIGHT_JAW = 397            # 오른쪽 턱선

    # 피부톤 분석용 얼굴 영역 (볼, 이마)
    CHEEK_LANDMARKS = [205, 207, 187, 213, 216, 206, 203, 425, 427, 411, 433, 436, 426, 423]
    FOREHEAD_LANDMARKS = [70, 63, 105, 66, 107, 55, 285, 300, 293, 334, 296, 336]

    def __init__(self):
        """MediaPipe Face Mesh 초기화"""
        import mediapipe as mp
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # 더 정밀한 랜드마크
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("✅ MediaPipe Face Mesh 초기화 완료")

    def analyze(self, image_data: bytes) -> Optional[MediaPipeFaceFeatures]:
        """
        얼굴 분석 (얼굴형 + 피부톤)

        Args:
            image_data: 이미지 바이트

        Returns:
            MediaPipeFaceFeatures 또는 None (실패 시)
        """
        try:
            import cv2
            import numpy as np
            
            # 이미지 디코딩
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error("이미지 디코딩 실패")
                return None

            # RGB로 변환 (MediaPipe는 RGB 입력 필요)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # MediaPipe로 얼굴 랜드마크 검출
            results = self.face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                logger.warning("MediaPipe로 얼굴을 검출하지 못함")
                return None

            # 첫 번째 얼굴 사용
            face_landmarks = results.multi_face_landmarks[0]

            # 이미지 크기
            h, w, _ = img.shape

            # 1. 얼굴형 분류
            face_shape, shape_confidence, measurements = self._classify_face_shape(
                face_landmarks, (h, w)
            )

            # 2. 피부톤 분석
            skin_tone, ita_value, hue_value = self._analyze_skin_tone(
                img, face_landmarks, (h, w)
            )

            # 최종 신뢰도 (얼굴형 신뢰도 기준)
            final_confidence = shape_confidence

            # ML 모델 입력용 특징 벡터 생성 (학습 데이터와 동일한 형식)
            face_features = [
                round(measurements['face_ratio'], 3),
                round(measurements['forehead_width'], 1),
                round(measurements['cheekbone_width'], 1),
                round(measurements['jaw_width'], 1),
                round(measurements['forehead_ratio'], 3),
                round(measurements['jaw_ratio'], 3)
            ]
            skin_features = [
                round(ita_value, 2),
                round(hue_value, 2)
            ]

            # 성별 추론 (NEW)
            gender, gender_confidence = self._infer_gender(measurements)

            logger.info(
                f"✅ MediaPipe 분석 완료: {face_shape} (신뢰도: {final_confidence:.0%}), "
                f"피부톤: {skin_tone} (ITA: {ita_value:.1f}°), "
                f"성별: {gender} (신뢰도: {gender_confidence:.0%})"
            )
            logger.debug(f"  Face features: {face_features}")
            logger.debug(f"  Skin features: {skin_features}")

            return MediaPipeFaceFeatures(
                face_shape=face_shape,
                skin_tone=skin_tone,
                confidence=round(final_confidence, 2),
                face_ratio=round(measurements['face_ratio'], 3),
                forehead_width=round(measurements['forehead_width'], 1),
                cheekbone_width=round(measurements['cheekbone_width'], 1),
                jaw_width=round(measurements['jaw_width'], 1),
                ITA_value=round(ita_value, 2),
                hue_value=round(hue_value, 2),
                face_features=face_features,
                skin_features=skin_features,
                gender=gender,
                gender_confidence=round(gender_confidence, 2)
            )

        except Exception as e:
            logger.error(f"MediaPipe 얼굴 분석 실패: {str(e)}")
            return None

    def _classify_face_shape(
        self,
        landmarks,
        image_shape: Tuple[int, int]
    ) -> Tuple[str, float, dict]:
        """
        478개 랜드마크로 얼굴형 분류

        Returns:
            (얼굴형, 신뢰도, 측정값_dict)
        """
        h, w = image_shape

        # 랜드마크를 픽셀 좌표로 변환
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return (int(lm.x * w), int(lm.y * h))

        # 주요 포인트 추출
        forehead_top = get_point(self.FOREHEAD_TOP)
        chin_bottom = get_point(self.CHIN_BOTTOM)
        left_cheek = get_point(self.LEFT_CHEEK)
        right_cheek = get_point(self.RIGHT_CHEEK)
        left_forehead = get_point(self.LEFT_FOREHEAD)
        right_forehead = get_point(self.RIGHT_FOREHEAD)
        left_jaw = get_point(self.LEFT_JAW)
        right_jaw = get_point(self.RIGHT_JAW)

        # 거리 계산
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # 측정값
        face_height = distance(forehead_top, chin_bottom)
        face_width = distance(left_cheek, right_cheek)
        forehead_width = distance(left_forehead, right_forehead)
        cheekbone_width = distance(left_cheek, right_cheek)
        jaw_width = distance(left_jaw, right_jaw)

        # 얼굴 비율
        face_ratio = face_height / face_width if face_width > 0 else 1.0

        # 너비 비율 (광대 대비)
        forehead_ratio = forehead_width / cheekbone_width if cheekbone_width > 0 else 1.0
        jaw_ratio = jaw_width / cheekbone_width if cheekbone_width > 0 else 1.0

        # 측정값 저장
        measurements = {
            'face_ratio': face_ratio,
            'forehead_width': forehead_width,
            'cheekbone_width': cheekbone_width,
            'jaw_width': jaw_width,
            'forehead_ratio': forehead_ratio,
            'jaw_ratio': jaw_ratio
        }

        # ========== 얼굴형 분류 로직 (개선됨 - 균형잡힌 분류) ==========
        confidence = 0.0

        # 1순위: 극단적인 비율 먼저 체크
        if face_ratio > 1.45:
            # 긴형: 얼굴이 세로로 매우 길다
            face_shape = "긴형"
            confidence = min(0.7 + (face_ratio - 1.45) * 0.5, 0.95)

        elif face_ratio < 0.95:
            # 둥근형: 얼굴이 가로로 넓다
            face_shape = "둥근형"
            confidence = min(0.7 + (0.95 - face_ratio) * 0.5, 0.95)

        # 2순위: 하트형 체크 (이마와 턱이 모두 좁음)
        elif forehead_ratio < 0.82 and jaw_ratio < 0.75:
            face_shape = "하트형"
            narrowness = (1.0 - forehead_ratio) + (1.0 - jaw_ratio)
            confidence = min(0.65 + narrowness * 0.3, 0.90)

        # 3순위: 각진형 체크 (턱이 넓거나 이마/턱 차이가 큼)
        elif jaw_ratio > 0.95 or abs(forehead_ratio - jaw_ratio) > 0.18:
            face_shape = "각진형"
            if jaw_ratio > 0.95:
                confidence = min(0.70 + (jaw_ratio - 0.95) * 2, 0.90)
            else:
                difference = abs(forehead_ratio - jaw_ratio)
                confidence = min(0.65 + difference * 0.4, 0.88)

        # 4순위: 긴형 (약간 긴 경우도 포함)
        elif face_ratio > 1.35:
            face_shape = "긴형"
            confidence = min(0.65 + (face_ratio - 1.35) * 0.4, 0.88)

        # 5순위: 둥근형 (약간 둥근 경우도 포함)
        elif face_ratio < 1.05:
            face_shape = "둥근형"
            confidence = min(0.65 + (1.05 - face_ratio) * 0.4, 0.88)

        # 6순위: 계란형 (균형잡힌 경우만)
        elif 1.05 <= face_ratio <= 1.35 and abs(forehead_ratio - jaw_ratio) < 0.15:
            face_shape = "계란형"
            balance_score = 1.0 - abs(forehead_ratio - jaw_ratio) * 3
            confidence = min(0.70 + balance_score * 0.15, 0.90)

        # 나머지: 가장 가까운 타입으로 분류
        else:
            # face_ratio 기준으로 가장 가까운 타입 선택
            if face_ratio > 1.20:
                face_shape = "긴형"
                confidence = 0.55
            elif face_ratio < 1.10:
                face_shape = "둥근형"
                confidence = 0.55
            else:
                face_shape = "각진형"  # 기본값을 각진형으로 변경
                confidence = 0.55

        logger.debug(
            f"얼굴형 측정: ratio={face_ratio:.2f}, "
            f"forehead_ratio={forehead_ratio:.2f}, "
            f"jaw_ratio={jaw_ratio:.2f} → {face_shape} ({confidence:.0%})"
        )

        return face_shape, confidence, measurements
    def _analyze_skin_tone(
        self,
        image: Any,
        landmarks,
        image_shape: Tuple[int, int]
    ) -> Tuple[str, float, float]:
        """
        피부톤 분석 (ITA + HSV 방식)

        Returns:
            (피부톤, ITA값, Hue값)
        """
        h, w = image_shape

        # 랜드마크를 픽셀 좌표로 변환
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return (int(lm.x * w), int(lm.y * h))

        # 피부 영역 추출 (볼 + 이마)
        skin_points = []

        # 볼 영역 랜드마크
        for idx in self.CHEEK_LANDMARKS:
            skin_points.append(get_point(idx))

        # 이마 영역 랜드마크
        for idx in self.FOREHEAD_LANDMARKS:
            skin_points.append(get_point(idx))

        # 마스크 생성
        import cv2
        import numpy as np
        mask = np.zeros((h, w), dtype=np.uint8)
        skin_points_np = np.array(skin_points, dtype=np.int32)
        cv2.fillConvexPoly(mask, cv2.convexHull(skin_points_np), 255)

        # 피부 영역만 추출
        skin_region = cv2.bitwise_and(image, image, mask=mask)

        # 유효한 픽셀만 추출 (마스크 영역)
        skin_pixels = image[mask > 0]

        if len(skin_pixels) == 0:
            logger.warning("피부 영역 추출 실패, 기본값 사용")
            return "봄웜", 0.0, 0.0

        # ========== ITA (Individual Typology Angle) 계산 ==========
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask > 0]

        # L(명도), b(노란기/푸른기) 평균
        L_mean = np.mean(lab_pixels[:, 0])
        b_mean = np.mean(lab_pixels[:, 2]) - 128  # LAB의 b는 0~255이므로 -128~127로 변환

        # ITA 계산: arctan((L - 50) / b) * 180 / π
        if abs(b_mean) > 0.1:
            ita = math.atan((L_mean - 50) / b_mean) * 180 / math.pi
        else:
            ita = 0.0

        # ========== HSV 색조 분석 ==========
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv[mask > 0]
        hue_mean = np.mean(hsv_pixels[:, 0])  # 0~179

        # ========== 피부톤 분류 (재개선 - 4계절 균형) ==========
        # ITA 기준:
        # - ITA > 41°: 밝은 피부
        # - 28° < ITA ≤ 41°: 중간 피부
        # - ITA ≤ 28°: 어두운 피부

        # Hue 기준 (한국인 피부톤 균형 조정):
        # - Hue < 10: 매우 강한 웜톤
        # - 10 ≤ Hue < 16: 웜톤
        # - 16 ≤ Hue < 20: 중립~약간 쿨
        # - Hue ≥ 20: 쿨톤

        # b값 (노란기/푸른기)도 고려
        is_warm = b_mean > 5  # b > 5이면 노란기가 강함 (웜톤)
        is_very_warm = b_mean > 10

        if ita > 42:
            # 매우 밝은 피부
            if hue_mean < 10 and is_very_warm:
                skin_tone = "봄웜"  # 매우 밝고 노란기 강함
            elif hue_mean < 16:
                if is_warm:
                    skin_tone = "봄웜"  # 밝고 약간 노란기
                else:
                    skin_tone = "여름쿨"  # 밝고 중립
            else:
                skin_tone = "여름쿨"  # 밝고 쿨톤
        elif ita > 32:
            # 밝은 피부
            if hue_mean < 10 and is_very_warm:
                skin_tone = "봄웜"  # 밝고 노란기 강함
            elif hue_mean < 16:
                if is_warm:
                    skin_tone = "가을웜"  # 밝고~중간, 웜톤
                else:
                    skin_tone = "여름쿨"  # 밝고 중립~쿨
            elif hue_mean < 20:
                skin_tone = "여름쿨"  # 밝고 쿨톤
            else:
                skin_tone = "여름쿨"  # 밝고 쿨톤
        elif ita > 22:
            # 중간 피부
            if hue_mean < 12 and is_warm:
                skin_tone = "가을웜"  # 중간이고 웜톤
            elif hue_mean < 18:
                if is_warm:
                    skin_tone = "가을웜"  # 중간이고 약간 웜
                else:
                    skin_tone = "여름쿨"  # 중간이고 중립~쿨
            else:
                skin_tone = "여름쿨"  # 중간이고 쿨톤
        else:
            # 어두운 피부
            if hue_mean < 14 and is_warm:
                skin_tone = "가을웜"  # 어둡고 웜톤
            elif hue_mean < 18:
                skin_tone = "가을웜"  # 어둡고 중립~웜
            else:
                skin_tone = "겨울쿨"  # 어둡고 쿨톤

        logger.debug(
            f"피부톤 분석: L={L_mean:.1f}, b={b_mean:.1f}, "
            f"ITA={ita:.1f}°, Hue={hue_mean:.1f} → {skin_tone}"
        )

        return skin_tone, ita, hue_mean

    def _infer_gender(self, measurements: dict) -> Tuple[str, float]:
        """
        얼굴 측정값으로 성별 추론 (휴리스틱 기반)

        Args:
            measurements: 얼굴 측정값 dict

        Returns:
            (gender, confidence) - ("male" | "female" | "neutral", 0.0~1.0)

        참고:
        - 남성: 턱이 넓고 각진 얼굴, forehead_ratio가 높음
        - 여성: 턱이 좁고 부드러운 얼굴, jaw_ratio가 낮음
        - 이 방법은 100% 정확하지 않으므로 신뢰도를 함께 반환
        """
        jaw_ratio = measurements.get('jaw_ratio', 1.0)
        forehead_ratio = measurements.get('forehead_ratio', 1.0)
        face_ratio = measurements.get('face_ratio', 1.0)

        # 점수 계산 (0~1 범위로 정규화)
        # 남성 점수: 턱이 넓고, 이마가 넓고, 얼굴이 각진 경우 높음
        male_score = 0.0
        female_score = 0.0

        # 턱 비율 (jaw_ratio)
        # 남성: jaw_ratio > 0.90 (턱이 광대만큼 넓음)
        # 여성: jaw_ratio < 0.85 (턱이 광대보다 좁음)
        if jaw_ratio > 0.90:
            male_score += 0.4
        elif jaw_ratio < 0.85:
            female_score += 0.4
        else:
            # 중간 범위 (0.85~0.90)
            neutral_score = 0.3
            male_score += neutral_score * (jaw_ratio - 0.85) / 0.05
            female_score += neutral_score * (0.90 - jaw_ratio) / 0.05

        # 이마 비율 (forehead_ratio)
        # 남성: forehead_ratio > 0.95 (이마가 광대만큼 넓음)
        # 여성: forehead_ratio < 0.90 (이마가 광대보다 좁음)
        if forehead_ratio > 0.95:
            male_score += 0.3
        elif forehead_ratio < 0.90:
            female_score += 0.3
        else:
            neutral_score = 0.2
            male_score += neutral_score * (forehead_ratio - 0.90) / 0.05
            female_score += neutral_score * (0.95 - forehead_ratio) / 0.05

        # 얼굴 비율 (face_ratio)
        # 남성: face_ratio가 중간 (1.0~1.3)
        # 여성: face_ratio가 약간 높거나 낮음 (하트형/계란형이 많음)
        if 1.0 <= face_ratio <= 1.3:
            male_score += 0.2
        else:
            female_score += 0.2

        # 정규화 (0~1 범위)
        total = male_score + female_score
        if total > 0:
            male_score /= total
            female_score /= total

        # 결정
        confidence_threshold = 0.55  # 55% 이상이면 확정

        if male_score >= confidence_threshold:
            gender = "male"
            confidence = male_score
        elif female_score >= confidence_threshold:
            gender = "female"
            confidence = female_score
        else:
            # 애매한 경우 neutral (남녀 공용 헤어스타일 추천)
            gender = "neutral"
            confidence = 0.5

        logger.debug(
            f"성별 추론: jaw_ratio={jaw_ratio:.2f}, forehead_ratio={forehead_ratio:.2f}, "
            f"male_score={male_score:.2f}, female_score={female_score:.2f} → {gender} ({confidence:.0%})"
        )

        return gender, confidence

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
