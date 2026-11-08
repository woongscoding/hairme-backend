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

    face_shape: str  # "계란형", "둥근형", "각진형", "긴형", "하트형"
    skin_tone: str  # "봄웜", "가을웜", "여름쿨", "겨울쿨"
    confidence: float  # 신뢰도 (0.0 ~ 1.0)

    # 측정값 (디버깅용)
    face_ratio: float  # 높이/너비
    forehead_width: float  # 이마 너비 (픽셀)
    cheekbone_width: float  # 광대 너비 (픽셀)
    jaw_width: float  # 턱 너비 (픽셀)
    # ⚠️ 레거시 스케일 ITA. OpenCV 8bit L(0~255)을 L*(0~100) 자리에 그대로 넣어 계산한
    # 값이라 실제 ITA보다 크게 나온다(대략 50~90°). 그래도 ML 추천 모델 입력(skin_features),
    # DynamoDB mediapipe_ITA_value, MLOps 학습 샘플, SKIN_FEATURE_STATS 정규화가 모두 이
    # 분포에 맞춰져 있어 바꾸면 학습/서빙 스큐가 생긴다. 피부톤 분류·밝기 판정은 ITA_corrected 를 쓴다.
    ITA_value: float  # 피부톤 ITA 값 (레거시 스케일, 모델 입력용)
    hue_value: float  # 색조 값 (HSV)

    # ML 모델 입력용 특징 벡터 (학습 데이터와 동일한 형식)
    face_features: list  # [face_ratio, forehead_width, cheekbone_width, jaw_width, forehead_ratio, jaw_ratio] (6차원)
    skin_features: list  # [ITA_value, hue_value] (2차원)

    # 성별 추론 (NEW)
    gender: str = "neutral"  # "male", "female", "neutral" (추론 실패 시)
    gender_confidence: float = 0.0  # 성별 추론 신뢰도

    # 고개 각도 (도). 좌우(yaw)·상하(pitch). 정면이면 0에 가깝다
    head_yaw: float = 0.0
    head_pitch: float = 0.0

    # 표준 ITA (L* = L_cv * 100/255 로 보정, 도). 피부톤 분류·사용자 표시용.
    # 모델 입력에는 쓰지 않는다. None 이면 미계산(테스트 픽스처 등).
    ITA_corrected: Optional[float] = None

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
            "gender_confidence": self.gender_confidence,
            "head_yaw": self.head_yaw,
            "head_pitch": self.head_pitch,
            "ITA_corrected": self.ITA_corrected,
        }


# ========== 피부톤 ITA 계산/분류 (순수 함수, 테스트 가능) ==========
# ITA 4단계 경계 (보정된 ITA 기준, 도)
SKIN_TONE_ITA_THRESHOLDS = (42.0, 32.0, 22.0)


def compute_ita(L_cv_mean: float, b_star_mean: float) -> Tuple[float, float]:
    """
    ITA (Individual Typology Angle) 계산.

    Args:
        L_cv_mean: OpenCV 8bit LAB 의 L 평균 (0~255)
        b_star_mean: b* 평균 (OpenCV b - 128, 약 -128~127)

    Returns:
        (legacy_ita, corrected_ita)
        - legacy_ita: 예전 공식 그대로 (L_cv 를 L* 로 착각). 모델 입력/DB 저장용으로 유지.
        - corrected_ita: L* = L_cv * 100/255 로 보정한 표준 ITA. 분류/표시용.
    """
    if abs(b_star_mean) <= 0.1:
        return 0.0, 0.0
    legacy = math.atan((L_cv_mean - 50) / b_star_mean) * 180 / math.pi
    L_star = L_cv_mean * 100.0 / 255.0
    corrected = math.atan((L_star - 50) / b_star_mean) * 180 / math.pi
    return legacy, corrected


def classify_skin_tone(ita: float, hue_mean: float, b_star_mean: float) -> str:
    """
    보정된 ITA + HSV Hue + b* 로 4계절 피부톤 분류.

    Args:
        ita: 보정된 ITA (compute_ita 의 corrected_ita)
        hue_mean: HSV Hue 평균 (OpenCV 0~179)
        b_star_mean: b* 평균
    """
    # Hue 기준 (한국인 피부톤 균형 조정):
    # - Hue < 10: 매우 강한 웜톤
    # - 10 ≤ Hue < 16: 웜톤
    # - 16 ≤ Hue < 20: 중립~약간 쿨
    # - Hue ≥ 20: 쿨톤
    # b값 (노란기/푸른기)도 고려
    is_warm = b_star_mean > 5  # b > 5이면 노란기가 강함 (웜톤)
    is_very_warm = b_star_mean > 10
    t_very_light, t_light, t_medium = SKIN_TONE_ITA_THRESHOLDS

    if ita > t_very_light:
        # 매우 밝은 피부
        if hue_mean < 10 and is_very_warm:
            return "봄웜"  # 매우 밝고 노란기 강함
        elif hue_mean < 16:
            return "봄웜" if is_warm else "여름쿨"  # 밝고 약간 노란기 / 밝고 중립
        return "여름쿨"  # 밝고 쿨톤
    elif ita > t_light:
        # 밝은 피부
        if hue_mean < 10 and is_very_warm:
            return "봄웜"  # 밝고 노란기 강함
        elif hue_mean < 16:
            return "가을웜" if is_warm else "여름쿨"  # 밝고~중간 웜 / 중립~쿨
        return "여름쿨"  # 밝고 쿨톤
    elif ita > t_medium:
        # 중간 피부
        if hue_mean < 12 and is_warm:
            return "가을웜"  # 중간이고 웜톤
        elif hue_mean < 18:
            return "가을웜" if is_warm else "여름쿨"  # 약간 웜 / 중립~쿨
        return "여름쿨"  # 중간이고 쿨톤
    else:
        # 어두운 피부
        if hue_mean < 14 and is_warm:
            return "가을웜"  # 어둡고 웜톤
        elif hue_mean < 18:
            return "가을웜"  # 어둡고 중립~웜
        return "겨울쿨"  # 어둡고 쿨톤


class MediaPipeFaceAnalyzer:
    """MediaPipe 기반 얼굴 분석기"""

    # MediaPipe 주요 랜드마크 인덱스
    # 얼굴 윤곽선 및 주요 포인트
    FOREHEAD_TOP = 10  # 이마 상단
    CHIN_BOTTOM = 152  # 턱 하단
    LEFT_CHEEK = 234  # 왼쪽 광대
    RIGHT_CHEEK = 454  # 오른쪽 광대
    LEFT_FOREHEAD = 70  # 왼쪽 이마 측면
    RIGHT_FOREHEAD = 300  # 오른쪽 이마 측면
    LEFT_JAW = 172  # 왼쪽 턱선
    RIGHT_JAW = 397  # 오른쪽 턱선

    # 피부톤 분석용 얼굴 영역 (볼, 이마)
    CHEEK_LANDMARKS = [
        205,
        207,
        187,
        213,
        216,
        206,
        203,
        425,
        427,
        411,
        433,
        436,
        426,
        423,
    ]
    FOREHEAD_LANDMARKS = [70, 63, 105, 66, 107, 55, 285, 300, 293, 334, 296, 336]

    def __init__(self):
        """MediaPipe Face Mesh 초기화"""
        import mediapipe as mp

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,  # 더 정밀한 랜드마크
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
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
            # ita_value 는 레거시 스케일(모델 입력용), ita_corrected 는 표준 ITA
            skin_tone, ita_value, hue_value, ita_corrected = self._analyze_skin_tone(
                img, face_landmarks, (h, w)
            )

            # 최종 신뢰도 (얼굴형 신뢰도 기준)
            final_confidence = shape_confidence

            # ML 모델 입력용 특징 벡터 생성 (학습 데이터와 동일한 형식)
            face_features = [
                round(measurements["face_ratio"], 3),
                round(measurements["forehead_width"], 1),
                round(measurements["cheekbone_width"], 1),
                round(measurements["jaw_width"], 1),
                round(measurements["forehead_ratio"], 3),
                round(measurements["jaw_ratio"], 3),
            ]
            # 레거시 스케일 ITA 유지: 학습 데이터·SKIN_FEATURE_STATS 와 같은 분포여야 한다
            skin_features = [round(ita_value, 2), round(hue_value, 2)]

            # 성별 추론 (NEW)
            gender, gender_confidence = self._infer_gender(measurements)

            logger.info(
                f"✅ MediaPipe 분석 완료: {face_shape} (신뢰도: {final_confidence:.0%}), "
                f"피부톤: {skin_tone} (ITA: {ita_corrected:.1f}°, "
                f"legacy {ita_value:.1f}), "
                f"성별: {gender} (신뢰도: {gender_confidence:.0%}), "
                f"고개 yaw={measurements['head_yaw']:.0f}° "
                f"pitch={measurements['head_pitch']:.0f}°"
            )
            logger.debug(f"  Face features: {face_features}")
            logger.debug(f"  Skin features: {skin_features}")

            return MediaPipeFaceFeatures(
                face_shape=face_shape,
                skin_tone=skin_tone,
                confidence=round(final_confidence, 2),
                face_ratio=round(measurements["face_ratio"], 3),
                forehead_width=round(measurements["forehead_width"], 1),
                cheekbone_width=round(measurements["cheekbone_width"], 1),
                jaw_width=round(measurements["jaw_width"], 1),
                ITA_value=round(ita_value, 2),
                hue_value=round(hue_value, 2),
                face_features=face_features,
                skin_features=skin_features,
                gender=gender,
                gender_confidence=round(gender_confidence, 2),
                head_yaw=round(measurements["head_yaw"], 1),
                head_pitch=round(measurements["head_pitch"], 1),
                ITA_corrected=round(ita_corrected, 2),
            )

        except Exception as e:
            logger.error(f"MediaPipe 얼굴 분석 실패: {str(e)}")
            return None

    # ========== 얼굴형 임계값 (실측 분포 기반, 2026-09-17 재보정) ==========
    # 이 클래스의 랜드마크 정의(턱 172/397, 광대 234/454, 이마 70/300)로 잰 비율은
    # 사람 간 편차가 매우 작다. 운영 DB에 저장된 측정값 11,599건(2026-03~09)의 분포:
    #   face_ratio     평균 1.228 ± 0.061 (p5 1.13 ~ p95 1.31)
    #   jaw_ratio      평균 0.783 ± 0.025 (p5 0.74 ~ p95 0.82)
    #   forehead_ratio 평균 0.845 ± 0.026 (p5 0.80 ~ p95 0.89)
    # 이전의 교과서식 절대 임계값(예: 각진형 jaw_ratio > 0.95)은 이 분포에서 도달이
    # 불가능해 10개월간 96.6%가 계란형으로 분류됐다. 아래 값은 위 분포의 분위수다.
    # 재도출: hairme-analysis 의 mediapipe_face_ratio·jaw_ratio·forehead_ratio 분위수.
    #
    # 2026-11-08 하트형 규칙 수정: 이전 규칙(이마·턱 모두 좁음)은 다이아몬드형 조건이었고,
    # 이마와 턱 폭은 서로 반대로 움직여(상관 -0.45) 1%만 걸렸다. 하트형은 넓은 이마 +
    # 좁은 턱이다. 측정값 13,943건(2025-11~2026-11)에 적용하면
    # 계란형 43% / 둥근형 21% / 긴형 18% / 각진형 11% / 하트형 9%.
    LONG_FACE_RATIO = 1.275  # p80 초과 → 긴형
    ROUND_FACE_RATIO = 1.185  # p20 미만 → 둥근형
    SQUARE_JAW_RATIO = 0.805  # p80 초과 → 각진형
    HEART_JAW_RATIO = 0.766  # p25 미만이고
    HEART_FOREHEAD_RATIO = 0.861  # p75 초과면 → 하트형
    # 신뢰도: 경계에서 0.60, 경계로부터 아래 폭만큼 멀어지면 0.90
    _FACE_RATIO_CONF_SPAN = 0.10
    _JAW_RATIO_CONF_SPAN = 0.04

    # 고개 각도: 측정은 3D 거리라 각도에 거의 영향받지 않지만, 많이 돌아가면 반대쪽
    # 윤곽 랜드마크가 가려져 추정이 부정확해진다. 이 각도를 넘으면 신뢰도를 낮춘다.
    POSE_LIMIT_DEG = 25.0
    POSE_LIMITED_CONFIDENCE = 0.5

    @classmethod
    def classify_from_ratios(
        cls, face_ratio: float, forehead_ratio: float, jaw_ratio: float
    ) -> Tuple[str, float]:
        """비율 3개로 얼굴형과 신뢰도를 결정한다 (순수 함수, 테스트 가능)."""

        def conf(distance: float, span: float) -> float:
            return round(min(0.60 + 0.30 * max(distance, 0.0) / span, 0.90), 2)

        # 1순위: 세로 비율 (가장 편차가 큰 측정값)
        if face_ratio > cls.LONG_FACE_RATIO:
            return "긴형", conf(
                face_ratio - cls.LONG_FACE_RATIO, cls._FACE_RATIO_CONF_SPAN
            )
        if face_ratio < cls.ROUND_FACE_RATIO:
            return "둥근형", conf(
                cls.ROUND_FACE_RATIO - face_ratio, cls._FACE_RATIO_CONF_SPAN
            )

        # 2순위: 이마는 넓고 턱은 좁음 → 하트형
        if (
            jaw_ratio < cls.HEART_JAW_RATIO
            and forehead_ratio > cls.HEART_FOREHEAD_RATIO
        ):
            margin = (cls.HEART_JAW_RATIO - jaw_ratio) + (
                forehead_ratio - cls.HEART_FOREHEAD_RATIO
            )
            return "하트형", conf(margin, cls._JAW_RATIO_CONF_SPAN)

        # 3순위: 턱이 넓음 → 각진형
        if jaw_ratio > cls.SQUARE_JAW_RATIO:
            return "각진형", conf(
                jaw_ratio - cls.SQUARE_JAW_RATIO, cls._JAW_RATIO_CONF_SPAN
            )

        # 나머지: 계란형. 구간 중앙에 가까울수록 신뢰도가 높다
        center = (cls.LONG_FACE_RATIO + cls.ROUND_FACE_RATIO) / 2
        half = (cls.LONG_FACE_RATIO - cls.ROUND_FACE_RATIO) / 2
        return "계란형", conf(half - abs(face_ratio - center), half)

    def _classify_face_shape(
        self, landmarks, image_shape: Tuple[int, int]
    ) -> Tuple[str, float, dict]:
        """
        478개 랜드마크로 얼굴형 분류

        Returns:
            (얼굴형, 신뢰도, 측정값_dict)
        """
        h, w = image_shape

        # 랜드마크를 픽셀 단위 3D 좌표로 변환. MediaPipe의 z는 x와 같은 축척
        # (이미지 너비 기준)이라 w를 곱한다.
        def get_point(idx):
            lm = landmarks.landmark[idx]
            return (lm.x * w, lm.y * h, lm.z * w)

        # 주요 포인트 추출
        forehead_top = get_point(self.FOREHEAD_TOP)
        chin_bottom = get_point(self.CHIN_BOTTOM)
        left_cheek = get_point(self.LEFT_CHEEK)
        right_cheek = get_point(self.RIGHT_CHEEK)
        left_forehead = get_point(self.LEFT_FOREHEAD)
        right_forehead = get_point(self.RIGHT_FOREHEAD)
        left_jaw = get_point(self.LEFT_JAW)
        right_jaw = get_point(self.RIGHT_JAW)

        # 3D 거리. 2D(사진 평면) 거리는 고개를 15° 돌리거나 숙이면 face_ratio가
        # ±0.04 변해 사람 간 차이(표준편차 0.06)와 맞먹었다. 3D 거리는 각도와 무관하고,
        # 정면 사진에서는 2D 거리와 같아(쓰는 점들이 정면에서 거의 같은 깊이) 기존
        # 임계값을 그대로 쓸 수 있다.
        distance = math.dist

        # 측정값
        face_height = distance(forehead_top, chin_bottom)
        face_width = distance(left_cheek, right_cheek)
        forehead_width = distance(left_forehead, right_forehead)
        cheekbone_width = distance(left_cheek, right_cheek)
        jaw_width = distance(left_jaw, right_jaw)

        head_yaw, head_pitch = self.estimate_head_pose(
            left_cheek, right_cheek, forehead_top, chin_bottom
        )

        # 얼굴 비율
        face_ratio = face_height / face_width if face_width > 0 else 1.0

        # 너비 비율 (광대 대비)
        forehead_ratio = (
            forehead_width / cheekbone_width if cheekbone_width > 0 else 1.0
        )
        jaw_ratio = jaw_width / cheekbone_width if cheekbone_width > 0 else 1.0

        # 측정값 저장
        measurements = {
            "face_ratio": face_ratio,
            "forehead_width": forehead_width,
            "cheekbone_width": cheekbone_width,
            "jaw_width": jaw_width,
            "forehead_ratio": forehead_ratio,
            "jaw_ratio": jaw_ratio,
            "head_yaw": head_yaw,
            "head_pitch": head_pitch,
        }

        face_shape, confidence = self.classify_from_ratios(
            face_ratio, forehead_ratio, jaw_ratio
        )
        confidence = self.limit_confidence_by_pose(confidence, head_yaw, head_pitch)

        logger.debug(
            f"얼굴형 측정: ratio={face_ratio:.2f}, "
            f"forehead_ratio={forehead_ratio:.2f}, "
            f"jaw_ratio={jaw_ratio:.2f}, yaw={head_yaw:.0f}°, "
            f"pitch={head_pitch:.0f}° → {face_shape} ({confidence:.0%})"
        )

        return face_shape, confidence, measurements

    @staticmethod
    def estimate_head_pose(
        left_cheek: Tuple[float, float, float],
        right_cheek: Tuple[float, float, float],
        forehead_top: Tuple[float, float, float],
        chin_bottom: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        """랜드마크 깊이(z)로 고개 각도(도)를 추정한다.

        yaw: 좌우 광대 점의 깊이 차이. 정면이면 대칭이라 0.
        pitch: 이마 꼭대기와 턱끝의 깊이 차이. 정면에서 두 점은 거의 같은 깊이다
        (MediaPipe 표준 얼굴 모델 기준 1° 미만).
        """
        yaw = math.degrees(
            math.atan2(right_cheek[2] - left_cheek[2], right_cheek[0] - left_cheek[0])
        )
        pitch = math.degrees(
            math.atan2(
                chin_bottom[2] - forehead_top[2], chin_bottom[1] - forehead_top[1]
            )
        )
        return yaw, pitch

    @classmethod
    def limit_confidence_by_pose(
        cls, confidence: float, head_yaw: float, head_pitch: float
    ) -> float:
        """고개가 많이 돌아간 사진은 신뢰도를 낮춘다 (판정 자체는 그대로)."""
        if max(abs(head_yaw), abs(head_pitch)) > cls.POSE_LIMIT_DEG:
            logger.info(
                f"고개 각도가 커서 얼굴형 신뢰도를 낮춤: yaw={head_yaw:.0f}°, "
                f"pitch={head_pitch:.0f}°"
            )
            return min(confidence, cls.POSE_LIMITED_CONFIDENCE)
        return confidence

    def _analyze_skin_tone(
        self, image: Any, landmarks, image_shape: Tuple[int, int]
    ) -> Tuple[str, float, float, float]:
        """
        피부톤 분석 (ITA + HSV 방식)

        Returns:
            (피부톤, 레거시 ITA값, Hue값, 보정 ITA값)
            레거시 ITA 는 모델 입력/DB 저장용, 분류는 보정 ITA 로 한다.
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
            return "봄웜", 0.0, 0.0, 0.0

        # ========== ITA (Individual Typology Angle) 계산 ==========
        # LAB 색공간으로 변환
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_pixels = lab[mask > 0]

        # L(명도), b(노란기/푸른기) 평균
        L_mean = np.mean(lab_pixels[:, 0])
        b_mean = (
            np.mean(lab_pixels[:, 2]) - 128
        )  # LAB의 b는 0~255이므로 -128~127로 변환

        # ITA 계산: arctan((L* - 50) / b*) * 180 / π
        # OpenCV 8bit LAB 의 L 은 0~255 라서 L* = L * 100/255 로 바꿔야 표준 ITA 가 된다.
        # 예전 공식(레거시)은 이 변환을 빠뜨려 ITA 가 항상 크게(50~90°) 나왔다.
        # 레거시 값은 모델 입력 호환 때문에 그대로 반환하고, 분류에는 보정값을 쓴다.
        ita_legacy, ita_corrected = compute_ita(float(L_mean), float(b_mean))

        # ========== HSV 색조 분석 ==========
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_pixels = hsv[mask > 0]
        hue_mean = np.mean(hsv_pixels[:, 0])  # 0~179

        # ========== 피부톤 분류 (보정 ITA 기준) ==========
        skin_tone = classify_skin_tone(ita_corrected, float(hue_mean), float(b_mean))

        logger.debug(
            f"피부톤 분석: L={L_mean:.1f}, b={b_mean:.1f}, "
            f"ITA={ita_corrected:.1f}° (legacy {ita_legacy:.1f}), "
            f"Hue={hue_mean:.1f} → {skin_tone}"
        )

        return skin_tone, ita_legacy, float(hue_mean), ita_corrected

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
        jaw_ratio = measurements.get("jaw_ratio", 1.0)
        forehead_ratio = measurements.get("forehead_ratio", 1.0)
        face_ratio = measurements.get("face_ratio", 1.0)

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
        if hasattr(self, "face_mesh"):
            self.face_mesh.close()
