"""얼굴형 임계값 재도출 (조회 전용).

2026-11-08 부터 얼굴 측정을 3D 거리로 바꿨다. 정면 사진은 값이 그대로지만, 고개 각도
때문에 생기던 흔들림이 빠져 분포가 좁아질 수 있다. 3D로 잰 행(mediapipe_head_yaw 가
있는 행)만 모아 분위수를 다시 계산하고, models/mediapipe_analyzer.py 의 현재 임계값과
그 임계값을 적용했을 때의 얼굴형 비율을 나란히 보여준다.

사용법:
    python scripts/derive_face_shape_thresholds.py           # 3D 측정 행만
    python scripts/derive_face_shape_thresholds.py --all     # 옛 2D 행 포함 (비교용)

출력의 "제안값"을 MediaPipeFaceAnalyzer 의 상수에 옮기고 tests/test_face_shape_*.py 를
돌리면 된다. 표본이 1,000건 미만이면 분위수가 흔들리므로 더 기다리는 게 낫다.
"""

import argparse
import collections
import sys
import types

import boto3
import numpy as np

TABLE = "hairme-analysis"
REGION = "ap-northeast-2"
MIN_ROWS = 1000

# 임계값 정의: (상수 이름, 측정값, 분위수)
QUANTILES = [
    ("ROUND_FACE_RATIO", "face", 20),
    ("LONG_FACE_RATIO", "face", 80),
    ("HEART_JAW_RATIO", "jaw", 25),
    ("SQUARE_JAW_RATIO", "jaw", 80),
    ("HEART_FOREHEAD_RATIO", "fore", 75),
]


def load_rows(include_2d: bool):
    table = boto3.resource("dynamodb", region_name=REGION).Table(TABLE)
    kwargs = {
        "ProjectionExpression": "created_at, gender, mediapipe_face_ratio, "
        "mediapipe_jaw_ratio, mediapipe_forehead_ratio, mediapipe_head_yaw, "
        "mediapipe_head_pitch"
    }
    rows = []
    while True:
        resp = table.scan(**kwargs)
        rows.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    need = ("mediapipe_face_ratio", "mediapipe_jaw_ratio", "mediapipe_forehead_ratio")
    rows = [r for r in rows if all(r.get(k) is not None for k in need)]
    if not include_2d:
        rows = [r for r in rows if r.get("mediapipe_head_yaw") is not None]
    return rows


def analyzer_class():
    # 분류 함수는 순수 함수라 mediapipe 없이 클래스만 가져온다
    for name in ("mediapipe", "mediapipe.solutions", "mediapipe.solutions.face_mesh"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, ".")
    from models.mediapipe_analyzer import MediaPipeFaceAnalyzer

    return MediaPipeFaceAnalyzer


def shape_share(cls, face, fore, jaw):
    counts = collections.Counter(
        cls.classify_from_ratios(float(f), float(o), float(j))[0]
        for f, o, j in zip(face, fore, jaw)
    )
    n = len(face)
    return ", ".join(f"{k} {v / n:.0%}" for k, v in counts.most_common())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="옛 2D 측정 행 포함")
    args = parser.parse_args()

    rows = load_rows(include_2d=args.all)
    label = "전체(2D+3D)" if args.all else "3D 측정"
    if not rows:
        print(f"{label} 행이 없습니다.")
        return
    dates = sorted(r["created_at"][:10] for r in rows)
    print(f"{label} {len(rows)}건 ({dates[0]} ~ {dates[-1]})")
    if len(rows) < MIN_ROWS:
        print(f"⚠️ {MIN_ROWS}건 미만이라 분위수가 불안정합니다. 참고만 하세요.")

    data = {
        "face": np.array([float(r["mediapipe_face_ratio"]) for r in rows]),
        "jaw": np.array([float(r["mediapipe_jaw_ratio"]) for r in rows]),
        "fore": np.array([float(r["mediapipe_forehead_ratio"]) for r in rows]),
    }
    for key, arr in data.items():
        print(f"  {key:5s} 평균 {arr.mean():.3f} ± {arr.std():.3f}")

    yaw = [
        abs(float(r["mediapipe_head_yaw"]))
        for r in rows
        if r.get("mediapipe_head_yaw") is not None
    ]
    if yaw:
        print(
            f"  고개 |yaw| 중앙값 {np.median(yaw):.1f}°, 25° 초과 {np.mean(np.array(yaw) > 25):.0%}"
        )

    cls = analyzer_class()
    print("\n상수                  현재    제안(분위수)")
    proposed = {}
    for name, key, q in QUANTILES:
        value = round(float(np.percentile(data[key], q)), 3)
        proposed[name] = value
        print(f"  {name:20s} {getattr(cls, name):.3f}  {value:.3f} (p{q})")

    print(
        "\n현재 임계값 적용 시: "
        + shape_share(cls, data["face"], data["fore"], data["jaw"])
    )
    for name, value in proposed.items():
        setattr(cls, name, value)
    print(
        "제안 임계값 적용 시: "
        + shape_share(cls, data["face"], data["fore"], data["jaw"])
    )

    for gender in ("male", "female"):
        idx = [i for i, r in enumerate(rows) if r.get("gender") == gender]
        if idx:
            print(
                f"  {gender} {len(idx)}건 (제안값): "
                + shape_share(
                    cls, data["face"][idx], data["fore"][idx], data["jaw"][idx]
                )
            )


if __name__ == "__main__":
    main()
