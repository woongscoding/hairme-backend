#!/usr/bin/env python3
"""
데이터셋 검증 스크립트

수집된 데이터의 품질과 분포를 확인합니다.

Author: HairMe ML Team
Date: 2025-11-15
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


def load_dataset(data_path: str) -> Dict:
    """NPZ 데이터셋 로드"""
    logger.info(f"📂 데이터 로딩: {data_path}")

    data = np.load(data_path, allow_pickle=False)

    face_features = data['face_features']  # [N, 6]
    skin_features = data['skin_features']  # [N, 2]
    hairstyles = data['hairstyles']  # [N]
    scores = data['scores']  # [N]

    # 메타데이터 (있는 경우)
    metadata = data.get('metadata', None)

    logger.info(f"✅ 데이터 로드 완료:")
    logger.info(f"  - 샘플 수: {len(scores):,}")
    logger.info(f"  - Face features: {face_features.shape}")
    logger.info(f"  - Skin features: {skin_features.shape}")
    logger.info(f"  - Hairstyles: {len(hairstyles)}")

    return {
        'face_features': face_features,
        'skin_features': skin_features,
        'hairstyles': hairstyles,
        'scores': scores,
        'metadata': metadata
    }


def validate_feature_ranges(data: Dict):
    """특징 값 범위 검증"""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 특징 값 범위 검증")
    logger.info("=" * 60)

    face_features = data['face_features']
    skin_features = data['skin_features']

    # Face features 검증
    face_names = [
        'face_ratio',
        'forehead_width',
        'cheekbone_width',
        'jaw_width',
        'forehead_ratio',
        'jaw_ratio'
    ]

    # 예상 범위 (MediaPipe 기준)
    expected_ranges = {
        'face_ratio': (0.8, 1.6),
        'forehead_width': (100, 300),
        'cheekbone_width': (120, 350),
        'jaw_width': (80, 280),
        'forehead_ratio': (0.7, 1.2),
        'jaw_ratio': (0.6, 1.1)
    }

    logger.info("\n📊 Face Features 범위:")
    for i, name in enumerate(face_names):
        values = face_features[:, i]
        min_val, max_val = expected_ranges[name]

        out_of_range = ((values < min_val) | (values > max_val)).sum()
        out_of_range_pct = out_of_range / len(values) * 100

        logger.info(f"  {name:20s}: [{values.min():7.2f}, {values.max():7.2f}]  "
                   f"(예상: [{min_val:7.2f}, {max_val:7.2f}])  "
                   f"범위 밖: {out_of_range:4d} ({out_of_range_pct:.1f}%)")

    # Skin features 검증
    logger.info("\n📊 Skin Features 범위:")

    ita_values = skin_features[:, 0]
    hue_values = skin_features[:, 1]

    logger.info(f"  ITA 피부톤:         [{ita_values.min():7.2f}, {ita_values.max():7.2f}]  "
               f"(예상: [-50, 70])")
    logger.info(f"  Hue:                [{hue_values.min():7.2f}, {hue_values.max():7.2f}]  "
               f"(예상: [0, 179])")

    # 이상치 감지
    logger.info("\n⚠️  이상치 확인:")

    # Face ratio가 너무 극단적인 경우
    extreme_ratio = ((face_features[:, 0] < 0.7) | (face_features[:, 0] > 1.7)).sum()
    if extreme_ratio > 0:
        logger.warning(f"  - 극단적 face_ratio: {extreme_ratio}개")

    # 너비 값이 비정상적인 경우
    for i, name in enumerate(['forehead_width', 'cheekbone_width', 'jaw_width']):
        idx = i + 1
        extreme = ((face_features[:, idx] < 50) | (face_features[:, idx] > 400)).sum()
        if extreme > 0:
            logger.warning(f"  - 극단적 {name}: {extreme}개")


def validate_score_distribution(data: Dict):
    """점수 분포 검증"""
    logger.info("\n" + "=" * 60)
    logger.info("📊 점수 분포 검증")
    logger.info("=" * 60)

    scores = data['scores']

    logger.info(f"\n기본 통계:")
    logger.info(f"  - 평균: {scores.mean():.2f}")
    logger.info(f"  - 표준편차: {scores.std():.2f}")
    logger.info(f"  - 중앙값: {np.median(scores):.2f}")
    logger.info(f"  - 최소값: {scores.min():.2f}")
    logger.info(f"  - 최대값: {scores.max():.2f}")

    # 점수 구간별 분포
    logger.info(f"\n점수 구간별 분포:")
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['매우 낮음 (0-20)', '낮음 (20-40)', '중간 (40-60)', '높음 (60-80)', '매우 높음 (80-100)']

    for i in range(len(bins) - 1):
        count = ((scores >= bins[i]) & (scores < bins[i+1])).sum()
        pct = count / len(scores) * 100
        logger.info(f"  {labels[i]:20s}: {count:5d}개 ({pct:5.1f}%)")

    # 균형 검증
    high_scores = (scores >= 80).sum()
    low_scores = (scores < 40).sum()
    high_pct = high_scores / len(scores) * 100
    low_pct = low_scores / len(scores) * 100

    logger.info(f"\n데이터 균형:")
    logger.info(f"  - 고점수 (≥80): {high_scores:,}개 ({high_pct:.1f}%)")
    logger.info(f"  - 저점수 (<40): {low_scores:,}개 ({low_pct:.1f}%)")

    if high_pct < 20 or low_pct < 20:
        logger.warning(f"  ⚠️ 데이터 불균형 감지! 학습에 영향을 줄 수 있습니다.")


def validate_hairstyle_distribution(data: Dict):
    """헤어스타일 분포 검증"""
    logger.info("\n" + "=" * 60)
    logger.info("💇 헤어스타일 분포 검증")
    logger.info("=" * 60)

    hairstyles = data['hairstyles']
    scores = data['scores']

    # 고유 헤어스타일 개수
    unique_styles = np.unique(hairstyles)
    logger.info(f"\n고유 헤어스타일: {len(unique_styles)}개")

    # 상위 20개 헤어스타일
    from collections import Counter
    style_counts = Counter(hairstyles.tolist())
    most_common = style_counts.most_common(20)

    logger.info(f"\n상위 20개 헤어스타일:")
    for rank, (style, count) in enumerate(most_common, 1):
        pct = count / len(hairstyles) * 100

        # 해당 스타일의 평균 점수
        style_scores = scores[hairstyles == style]
        avg_score = style_scores.mean()

        logger.info(f"  {rank:2d}. {style:30s}: {count:4d}개 ({pct:4.1f}%)  평균점수: {avg_score:.1f}")

    # 저빈도 스타일 (3개 이하)
    low_freq_styles = [style for style, count in style_counts.items() if count <= 3]
    if low_freq_styles:
        logger.warning(f"\n⚠️ 저빈도 스타일 (≤3개): {len(low_freq_styles)}개")


def validate_metadata(data: Dict):
    """메타데이터 검증"""
    if data['metadata'] is None:
        logger.info("\n⚠️ 메타데이터 없음")
        return

    logger.info("\n" + "=" * 60)
    logger.info("📋 메타데이터 검증")
    logger.info("=" * 60)

    metadata = data['metadata']

    # Face shape 분포
    face_shapes = [m['face_shape'] for m in metadata]
    from collections import Counter
    face_shape_counts = Counter(face_shapes)

    logger.info(f"\n얼굴형 분포:")
    for shape, count in face_shape_counts.most_common():
        pct = count / len(face_shapes) * 100
        logger.info(f"  {shape:10s}: {count:4d}개 ({pct:5.1f}%)")

    # Skin tone 분포
    skin_tones = [m['skin_tone'] for m in metadata]
    skin_tone_counts = Counter(skin_tones)

    logger.info(f"\n피부톤 분포:")
    for tone, count in skin_tone_counts.most_common():
        pct = count / len(skin_tones) * 100
        logger.info(f"  {tone:10s}: {count:4d}개 ({pct:5.1f}%)")

    # 신뢰도 분포
    confidences = [m['confidence'] for m in metadata]
    avg_conf = np.mean(confidences)
    logger.info(f"\nMediaPipe 신뢰도:")
    logger.info(f"  - 평균: {avg_conf:.1%}")
    logger.info(f"  - 최소: {min(confidences):.1%}")
    logger.info(f"  - 최대: {max(confidences):.1%}")

    low_conf_count = sum(1 for c in confidences if c < 0.8)
    if low_conf_count > 0:
        logger.warning(f"  ⚠️ 낮은 신뢰도 (<80%): {low_conf_count}개 얼굴")


def save_validation_report(data: Dict, output_path: str):
    """검증 리포트 저장"""
    logger.info(f"\n📄 검증 리포트 저장: {output_path}")

    report = {
        'total_samples': len(data['scores']),
        'face_features_shape': data['face_features'].shape,
        'skin_features_shape': data['skin_features'].shape,
        'score_stats': {
            'mean': float(data['scores'].mean()),
            'std': float(data['scores'].std()),
            'min': float(data['scores'].min()),
            'max': float(data['scores'].max()),
            'median': float(np.median(data['scores']))
        },
        'unique_hairstyles': len(np.unique(data['hairstyles'])),
        'has_metadata': data['metadata'] is not None
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("✅ 리포트 저장 완료")


def main():
    parser = argparse.ArgumentParser(description="데이터셋 검증")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="검증할 NPZ 데이터 경로"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_report.json",
        help="검증 리포트 저장 경로"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("🔍 데이터셋 검증 시작")
    logger.info("=" * 60)

    # 데이터 로드
    data = load_dataset(args.data)

    # 검증 수행
    validate_feature_ranges(data)
    validate_score_distribution(data)
    validate_hairstyle_distribution(data)
    validate_metadata(data)

    # 리포트 저장
    save_validation_report(data, args.output)

    logger.info("\n" + "=" * 60)
    logger.info("✅ 데이터셋 검증 완료!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
