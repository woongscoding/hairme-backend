#!/usr/bin/env python3
"""
여러 NPZ 파일을 하나로 병합

사용 예시:
  python merge_npz_files.py batch1.npz batch2.npz -o combined.npz
  python merge_npz_files.py data_source/ai_face_batch*.npz -o data_source/combined.npz

Author: HairMe ML Team
Date: 2025-11-15
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_npz_files(input_files, output_file):
    """
    여러 NPZ 파일을 병합

    Args:
        input_files: 입력 NPZ 파일 경로 리스트
        output_file: 출력 NPZ 파일 경로
    """
    logger.info(f"🔄 {len(input_files)}개 파일 병합 시작...")

    all_face_features = []
    all_skin_features = []
    all_hairstyles = []
    all_scores = []
    all_metadata = []

    total_samples = 0

    for i, file_path in enumerate(input_files, 1):
        logger.info(f"[{i}/{len(input_files)}] 로딩: {file_path}")

        try:
            data = np.load(file_path, allow_pickle=False)

            # 데이터 추출
            face_features = data["face_features"]
            skin_features = data["skin_features"]
            hairstyles = data["hairstyles"]
            scores = data["scores"]
            metadata = data["metadata"]

            # 리스트에 추가
            all_face_features.append(face_features)
            all_skin_features.append(skin_features)
            all_hairstyles.append(hairstyles)
            all_scores.append(scores)
            all_metadata.append(metadata)

            sample_count = len(scores)
            total_samples += sample_count

            logger.info(f"  ✅ {sample_count}개 샘플 추가 (누적: {total_samples}개)")

        except Exception as e:
            logger.error(f"  ❌ 파일 로드 실패: {str(e)}")
            continue

    if not all_scores:
        logger.error("❌ 병합할 데이터가 없습니다!")
        return

    # NumPy 배열로 병합
    logger.info("📊 데이터 병합 중...")

    merged_data = {
        "face_features": np.concatenate(all_face_features, axis=0),
        "skin_features": np.concatenate(all_skin_features, axis=0),
        "hairstyles": np.concatenate(all_hairstyles, axis=0),
        "scores": np.concatenate(all_scores, axis=0),
        "metadata": np.concatenate(all_metadata, axis=0),
    }

    # 저장
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_file, **merged_data)

    logger.info(f"\n{'='*60}")
    logger.info("✅ 병합 완료!")
    logger.info(f"{'='*60}")
    logger.info(f"  출력 파일: {output_file}")
    logger.info(f"  총 샘플: {len(merged_data['scores'])}개")
    logger.info(f"  Face features: {merged_data['face_features'].shape}")
    logger.info(f"  Skin features: {merged_data['skin_features'].shape}")
    logger.info(f"  Hairstyles: {len(merged_data['hairstyles'])}")
    logger.info(f"  Scores: {merged_data['scores'].shape}")
    logger.info(f"{'='*60}")

    # 통계 출력
    scores = merged_data["scores"]
    logger.info("\n📊 점수 통계:")
    logger.info(f"  - 추천 (≥70): {(scores >= 70).sum()}개")
    logger.info(f"  - 비추천 (<70): {(scores < 70).sum()}개")
    logger.info(f"  - 평균: {scores.mean():.1f}")
    logger.info(f"  - 표준편차: {scores.std():.1f}")

    # 얼굴형 분포
    from collections import Counter

    metadata = merged_data["metadata"]
    face_shapes = [m["face_shape"] for m in metadata]

    logger.info("\n📊 얼굴형 분포:")
    for shape, count in Counter(face_shapes).most_common():
        logger.info(f"  - {shape}: {count}개 ({count/len(face_shapes)*100:.1f}%)")

    # 피부톤 분포
    skin_tones = [m["skin_tone"] for m in metadata]
    logger.info("\n📊 피부톤 분포:")
    for tone, count in Counter(skin_tones).most_common():
        logger.info(f"  - {tone}: {count}개 ({count/len(skin_tones)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="NPZ 파일 병합")
    parser.add_argument(
        "input_files", nargs="+", help="입력 NPZ 파일들 (glob 패턴 사용 가능)"
    )
    parser.add_argument("-o", "--output", required=True, help="출력 NPZ 파일 경로")

    args = parser.parse_args()

    # Glob 패턴 확장
    input_files = []
    for pattern in args.input_files:
        # Glob 패턴인지 확인
        if "*" in pattern or "?" in pattern:
            matched = glob.glob(pattern)
            input_files.extend(matched)
        else:
            input_files.append(pattern)

    # 중복 제거 및 정렬
    input_files = sorted(set(input_files))

    if not input_files:
        logger.error("❌ 입력 파일이 없습니다!")
        return

    logger.info(f"📂 입력 파일 {len(input_files)}개:")
    for f in input_files:
        logger.info(f"  - {f}")
    logger.info("")

    # 병합 실행
    merge_npz_files(input_files, args.output)


if __name__ == "__main__":
    main()
