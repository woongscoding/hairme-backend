# models/body_pose_analyzer.py
"""
MediaPipe Pose 기반 전신 비율 분석 (체형별 아이템 추천용)

- 이미지 → 관절 좌표(BodyPoseAnalyzer.extract_landmarks)와
  좌표 → 비율 → 체형(compute_body_ratios / classify_body_from_ratios)을 분리한다.
  뒤의 둘은 순수 함수라 모델 없이 테스트할 수 있다.
- 사진은 메모리에서만 처리하고 어디에도 저장하지 않는다.
- 체중·체지방 같은 판정은 하지 않는다. 관절 사이 비율만 본다.
"""

# import cv2  # Lazy loaded
# import numpy as np  # Lazy loaded
# import mediapipe as mp  # Lazy loaded
import logging
import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from core.exceptions import FullBodyNotDetectedException, NotFrontalPoseException

logger = logging.getLogger(__name__)

# (x, y, visibility) - x/y 는 이미지 크기로 정규화된 0~1 좌표
Landmark = Tuple[float, float, float]

# MediaPipe Pose 랜드마크 인덱스
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
# 체형(어깨/골반 비율) 판정에 꼭 필요한 관절. 발이 잘리거나 무릎 위까지만 나온
# 사진도 받는다 - 실사용 첫날 거절된 사진이 대부분 발목만 살짝 잘린 경우였다.
REQUIRED_LANDMARKS = (
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
)
# 다리 비율(보조 표시 long_torso)용. 하나라도 안 보이면 다리 비율만 건너뛴다
LEG_LANDMARKS = (NOSE, LEFT_ANKLE, RIGHT_ANKLE)

# 이 값 미만이면 해당 관절이 사진에 안 보이는 것으로 본다
MIN_VISIBILITY = 0.5
# 정규화 좌표가 프레임 밖으로 이만큼 넘어가면 잘린 것으로 본다
# (MediaPipe 는 프레임 밖 관절도 추정해 내놓는다)
FRAME_MARGIN = 0.02
# 디코딩 후 긴 변 상한. 모델 입력은 256px 이라 큰 원본은 메모리만 쓴다
MAX_IMAGE_SIDE = 1280

# ========== 체형 임계값 (잠정 - 실사진 보정 전) ==========
# MediaPipe 11/12, 23/24 는 윤곽이 아니라 관절 위치다. 고관절 폭은 실제 골반 폭보다
# 훨씬 좁아서 어깨/골반 관절 폭 비율은 대부분 1.3~1.6 에 몰린다.
# 아래 값은 실측 분포가 아니라 가정한 분포(남 1.50±0.08, 여 1.38±0.08 - 여성의
# 골반 관절 간격이 상대적으로 넓다)의 p20/p80 이다. 얼굴형이 교과서식 절대값 때문에
# 96%가 한 유형으로 몰렸던 전례가 있어, 중앙 60%를 균형형으로 두는 분위수 방식을 따른다.
# 재보정: body_analysis_complete 로그의 shoulder_hip_ratio 성별 분위수로 다시 정한다.
BROAD_SHOULDER_RATIO = {"male": 1.57, "female": 1.45}  # 초과 → 어깨 넓은형
LOWER_VOLUME_RATIO = {"male": 1.43, "female": 1.31}  # 미만 → 하체 볼륨형

# 다리 비율 = 골반 중점→발목 중점 길이 / 코→발목 중점 길이.
# 신체 치수 기준 추정 평균은 0.52~0.55 지만 촬영 높이·원근에 민감하다.
# 보조 표시라 오탐이 적도록 추정 분포의 아래쪽으로 잡았다 (잠정).
LONG_TORSO_LEG_RATIO = 0.51  # 미만 → 상체 긴 비율

# 정면 판정: 어깨 폭 / 몸통 길이(어깨 중점→골반 중점).
# 정면은 0.6~0.8, 몸을 50도 이상 틀면 이 값 아래로 떨어진다 (잠정).
MIN_SHOULDER_TORSO_RATIO = 0.45

BODY_TYPES = ("broad_shoulder", "lower_volume", "balanced")
LONG_TORSO = "long_torso"


@dataclass(frozen=True)
class BodyRatios:
    """체형 분류 입력 비율"""

    shoulder_hip_ratio: float  # 어깨 관절 폭 / 골반 관절 폭
    # 다리 길이 / 코→발목 길이. 발목(또는 코)이 사진에 없으면 None
    leg_ratio: Optional[float]
    shoulder_torso_ratio: float  # 어깨 폭 / 몸통 길이 (정면 판정용)

    def to_dict(self) -> dict:
        return {
            "shoulder_hip": round(self.shoulder_hip_ratio, 3),
            "leg": None if self.leg_ratio is None else round(self.leg_ratio, 3),
            "shoulder_torso": round(self.shoulder_torso_ratio, 3),
        }


def _is_visible(landmark: Landmark) -> bool:
    """관절이 보이고 프레임 안에 있는지 (MediaPipe 는 프레임 밖 관절도 추정해 내놓는다)"""
    x, y, visibility = landmark
    return (
        visibility >= MIN_VISIBILITY
        and -FRAME_MARGIN <= x <= 1 + FRAME_MARGIN
        and -FRAME_MARGIN <= y <= 1 + FRAME_MARGIN
    )


def compute_body_ratios(
    landmarks: Sequence[Landmark], width: int, height: int
) -> BodyRatios:
    """관절 좌표로 체형 비율을 계산한다 (순수 함수).

    Raises:
        FullBodyNotDetectedException: 어깨·골반이 안 보이거나 프레임 밖
        NotFrontalPoseException: 몸이 옆으로 돌아가 비율을 잴 수 없음
    """
    if len(landmarks) <= max(REQUIRED_LANDMARKS + LEG_LANDMARKS):
        raise FullBodyNotDetectedException()
    if not all(_is_visible(landmarks[idx]) for idx in REQUIRED_LANDMARKS):
        raise FullBodyNotDetectedException()

    # 원본 화면비를 살린 픽셀 좌표에서 잰다 (정규화 좌표는 가로세로 축척이 다르다)
    def point(idx: int) -> Tuple[float, float]:
        return landmarks[idx][0] * width, landmarks[idx][1] * height

    def midpoint(a: int, b: int) -> Tuple[float, float]:
        (ax, ay), (bx, by) = point(a), point(b)
        return (ax + bx) / 2, (ay + by) / 2

    shoulder_width = math.dist(point(LEFT_SHOULDER), point(RIGHT_SHOULDER))
    hip_width = math.dist(point(LEFT_HIP), point(RIGHT_HIP))
    shoulder_mid = midpoint(LEFT_SHOULDER, RIGHT_SHOULDER)
    hip_mid = midpoint(LEFT_HIP, RIGHT_HIP)
    torso_length = math.dist(shoulder_mid, hip_mid)

    if min(hip_width, torso_length) <= 0:
        raise FullBodyNotDetectedException()

    shoulder_torso_ratio = shoulder_width / torso_length
    if shoulder_torso_ratio < MIN_SHOULDER_TORSO_RATIO:
        raise NotFrontalPoseException()

    leg_ratio: Optional[float] = None
    if all(_is_visible(landmarks[idx]) for idx in LEG_LANDMARKS):
        ankle_mid = midpoint(LEFT_ANKLE, RIGHT_ANKLE)
        body_length = math.dist(point(NOSE), ankle_mid)
        if body_length > 0:
            leg_ratio = math.dist(hip_mid, ankle_mid) / body_length

    return BodyRatios(
        shoulder_hip_ratio=shoulder_width / hip_width,
        leg_ratio=leg_ratio,
        shoulder_torso_ratio=shoulder_torso_ratio,
    )


def classify_body_from_ratios(
    shoulder_hip_ratio: float, leg_ratio: Optional[float], gender: str
) -> Tuple[str, List[str]]:
    """비율로 체형과 보조 비율 표시를 정한다 (순수 함수, 테스트 가능).

    Returns:
        (body_type, flags) - body_type 은 BODY_TYPES 중 하나,
        flags 는 보조 표시 목록 (현재는 "long_torso" 뿐)
    """
    key = "female" if gender == "female" else "male"

    if shoulder_hip_ratio > BROAD_SHOULDER_RATIO[key]:
        body_type = "broad_shoulder"
    elif shoulder_hip_ratio < LOWER_VOLUME_RATIO[key]:
        body_type = "lower_volume"
    else:
        body_type = "balanced"

    # 발목이 안 보여 다리 비율을 못 쟀으면 보조 표시는 붙이지 않는다
    flags = (
        [LONG_TORSO]
        if leg_ratio is not None and leg_ratio < LONG_TORSO_LEG_RATIO
        else []
    )
    return body_type, flags


class BodyPoseAnalyzer:
    """MediaPipe Pose 래퍼 - 이미지에서 관절 좌표만 뽑는다"""

    def __init__(self):
        """MediaPipe Pose 초기화"""
        import mediapipe as mp

        # model_complexity=1(full) 모델만 휠에 포함돼 있다. 0/2 는 첫 실행 때
        # 패키지 경로로 내려받는데 Lambda 는 /tmp 외 읽기 전용이라 실패한다.
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )
        # MediaPipe 그래프는 동시 호출에 안전하지 않다 (스레드풀에서 호출됨)
        self._lock = threading.Lock()
        logger.info("✅ MediaPipe Pose 초기화 완료")

    def extract_landmarks(
        self, image_data: bytes
    ) -> Optional[Tuple[List[Landmark], int, int]]:
        """
        이미지에서 전신 관절 좌표 추출

        Returns:
            (landmarks, width, height) 또는 None (디코딩 실패/사람 미검출)
        """
        import cv2
        import numpy as np

        # IMREAD_COLOR 는 JPEG EXIF 회전을 반영한다 (세로 전신 사진)
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            logger.error("이미지 디코딩 실패")
            return None

        h, w = img.shape[:2]
        scale = MAX_IMAGE_SIDE / max(h, w)
        if scale < 1:
            img = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            h, w = img.shape[:2]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with self._lock:
            results = self.pose.process(img_rgb)

        if not results.pose_landmarks:
            logger.warning("MediaPipe Pose 로 사람을 검출하지 못함")
            return None

        landmarks = [
            (lm.x, lm.y, lm.visibility) for lm in results.pose_landmarks.landmark
        ]
        return landmarks, w, h

    def __del__(self):
        """리소스 정리"""
        if hasattr(self, "pose"):
            self.pose.close()


_body_pose_analyzer: Optional[BodyPoseAnalyzer] = None
_init_lock = threading.Lock()


def get_body_pose_analyzer() -> BodyPoseAnalyzer:
    """BodyPoseAnalyzer 싱글톤 (첫 요청에서 지연 로딩 - 콜드스타트 영향 없음)"""
    global _body_pose_analyzer
    if _body_pose_analyzer is None:
        with _init_lock:
            if _body_pose_analyzer is None:
                logger.info("🐢 Lazy initializing BodyPoseAnalyzer...")
                _body_pose_analyzer = BodyPoseAnalyzer()
    return _body_pose_analyzer
