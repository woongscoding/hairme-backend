"""고개 각도에 따른 얼굴형 측정 안정성 테스트 (models/mediapipe_analyzer.py).

MediaPipe 표준 얼굴 모델(canonical_face_model.obj)의 실제 좌표를 돌려서 사진처럼
투영한 뒤 측정한다. 옛 2D 거리 방식은 고개를 15° 돌리면 face_ratio가 0.04 변해
사람 간 차이(표준편차 0.06)와 맞먹었다. 3D 거리는 각도와 무관해야 한다.
"""

import math
import sys
import types
from types import SimpleNamespace

import pytest

# canonical_face_model.obj 의 해당 정점 (cm, y 위쪽, z 카메라 쪽)
CANONICAL = {
    10: (0.0, 8.262, 4.482),  # 이마 꼭대기
    152: (0.0, -9.403, 4.264),  # 턱끝
    234: (-7.664, 0.673, -2.436),  # 왼쪽 광대(윤곽)
    454: (7.664, 0.673, -2.436),  # 오른쪽 광대(윤곽)
    70: (-5.721, 4.255, 2.831),  # 왼쪽 이마
    300: (5.721, 4.255, 2.831),  # 오른쪽 이마
    172: (-5.941, -6.224, -0.631),  # 왼쪽 턱선
    397: (5.941, -6.224, -0.631),  # 오른쪽 턱선
}
CANONICAL_FACE_RATIO = 1.153  # 정면에서의 face_ratio
IMG_H, IMG_W = 1280, 960


@pytest.fixture(scope="module")
def analyzer():
    """mediapipe 없이 분석기를 만든다 (__init__ 은 FaceMesh 를 띄우므로 건너뜀)."""
    for name in ("mediapipe", "mediapipe.solutions", "mediapipe.solutions.face_mesh"):
        sys.modules.setdefault(name, types.ModuleType(name))
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer

    return object.__new__(MediaPipeFaceAnalyzer)


def fake_landmarks(yaw_deg: float = 0.0, pitch_deg: float = 0.0, scale: float = 25.0):
    """표준 얼굴을 돌린 뒤 MediaPipe 정규화 좌표(x/w, y/h, z/w, y 아래쪽)로 만든다."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    landmark = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(468)]
    for idx, (X, Y, Z) in CANONICAL.items():
        # yaw(세로축) → pitch(가로축) 순서로 회전
        X, Z = X * math.cos(y) + Z * math.sin(y), -X * math.sin(y) + Z * math.cos(y)
        Y, Z = Y * math.cos(p) - Z * math.sin(p), Y * math.sin(p) + Z * math.cos(p)
        landmark[idx] = SimpleNamespace(
            x=(IMG_W / 2 + scale * X) / IMG_W,
            y=(IMG_H / 2 - scale * Y) / IMG_H,
            z=(-scale * Z) / IMG_W,
        )
    return SimpleNamespace(landmark=landmark)


def measure(analyzer, yaw=0.0, pitch=0.0):
    _, conf, m = analyzer._classify_face_shape(
        fake_landmarks(yaw, pitch), (IMG_H, IMG_W)
    )
    return conf, m


def test_frontal_matches_2d_so_old_thresholds_still_apply(analyzer):
    """정면 사진에선 3D 측정이 옛 2D 측정과 같아야 기존 임계값을 그대로 쓸 수 있다."""
    _, m = measure(analyzer)
    assert m["face_ratio"] == pytest.approx(CANONICAL_FACE_RATIO, abs=0.002)
    assert m["head_yaw"] == pytest.approx(0.0, abs=0.5)
    assert m["head_pitch"] == pytest.approx(0.0, abs=1.0)


@pytest.mark.parametrize(
    "yaw,pitch", [(15, 0), (-15, 0), (0, 15), (0, -15), (20, 10), (-10, -20)]
)
def test_ratios_do_not_change_with_head_pose(analyzer, yaw, pitch):
    _, front = measure(analyzer)
    _, turned = measure(analyzer, yaw, pitch)
    for key in ("face_ratio", "forehead_ratio", "jaw_ratio"):
        assert turned[key] == pytest.approx(front[key], abs=0.002), key


@pytest.mark.parametrize("yaw,pitch", [(20, 0), (-20, 0), (0, 15), (0, -15)])
def test_pose_is_estimated(analyzer, yaw, pitch):
    _, m = measure(analyzer, yaw, pitch)
    assert abs(m["head_yaw"]) == pytest.approx(abs(yaw), abs=1.0)
    assert abs(m["head_pitch"]) == pytest.approx(abs(pitch), abs=1.5)
    # 좌우가 바뀌면 부호가 바뀐다
    if yaw:
        _, mirrored = measure(analyzer, -yaw, pitch)
        assert math.copysign(1, mirrored["head_yaw"]) != math.copysign(1, m["head_yaw"])


def test_large_pose_limits_confidence(analyzer):
    cls = type(analyzer)
    conf_front, _ = measure(analyzer)
    conf_turned, _ = measure(analyzer, yaw=35)
    assert conf_turned <= cls.POSE_LIMITED_CONFIDENCE
    assert cls.limit_confidence_by_pose(0.9, 10, -10) == 0.9
    assert cls.limit_confidence_by_pose(0.9, 0, 30) == cls.POSE_LIMITED_CONFIDENCE
    assert cls.limit_confidence_by_pose(0.4, 40, 0) == 0.4  # 이미 낮으면 그대로
    assert conf_front >= 0.6
