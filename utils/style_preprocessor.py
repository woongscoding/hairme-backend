"""
헤어스타일명 전처리 유틸리티

띄어쓰기를 제거하여 일관성 있는 스타일명 생성
- "시스루 뱅" → "시스루뱅"
- "레이어드 컷" → "레이어드컷"
- "C컬 단발" → "C컬단발"

Author: HairMe ML Team
Date: 2025-11-11
Version: 1.0.0
"""

import re
from typing import List, Dict


def normalize_style_name(style_name: str) -> str:
    """
    헤어스타일명 정규화 (띄어쓰기 제거)

    Args:
        style_name: 원본 스타일명

    Returns:
        정규화된 스타일명 (띄어쓰기 제거)

    Examples:
        >>> normalize_style_name("시스루 뱅")
        "시스루뱅"

        >>> normalize_style_name("레이어드 컷")
        "레이어드컷"

        >>> normalize_style_name("  단발   보브  ")
        "단발보브"
    """
    if not style_name:
        return ""

    # 앞뒤 공백 제거 후 모든 띄어쓰기 제거
    normalized = style_name.strip().replace(" ", "")

    return normalized


def normalize_style_list(styles: List[str]) -> List[str]:
    """
    여러 스타일명을 일괄 정규화

    Args:
        styles: 스타일명 리스트

    Returns:
        정규화된 스타일명 리스트
    """
    return [normalize_style_name(s) for s in styles]


def normalize_training_data(data: Dict) -> Dict:
    """
    학습 데이터의 모든 스타일명을 정규화

    Args:
        data: 학습 데이터 딕셔너리

    Returns:
        정규화된 학습 데이터
    """
    normalized_data = {
        "metadata": data["metadata"].copy(),
        "statistics": data["statistics"].copy() if "statistics" in data else {},
        "training_data": []
    }

    # 각 이미지의 조합에서 스타일명 정규화
    for image_data in data["training_data"]:
        normalized_image = {
            "image_id": image_data["image_id"],
            "face_shape": image_data["face_shape"],
            "skin_tone": image_data["skin_tone"],
            "combinations": []
        }

        for combo in image_data["combinations"]:
            normalized_combo = combo.copy()
            # 스타일명 정규화
            normalized_combo["hairstyle"] = normalize_style_name(combo["hairstyle"])
            normalized_image["combinations"].append(normalized_combo)

        normalized_data["training_data"].append(normalized_image)

    return normalized_data


def get_unique_styles(data: Dict) -> List[str]:
    """
    학습 데이터에서 고유한 스타일명 추출 (정규화 후)

    Args:
        data: 학습 데이터 딕셔너리

    Returns:
        고유 스타일명 리스트 (정렬됨)
    """
    styles = set()

    for image_data in data["training_data"]:
        for combo in image_data["combinations"]:
            normalized = normalize_style_name(combo["hairstyle"])
            styles.add(normalized)

    return sorted(list(styles))


if __name__ == "__main__":
    # 테스트
    test_cases = [
        "시스루 뱅",
        "레이어드 컷",
        "C컬 단발",
        "단발 보브",
        "  허쉬컷  ",
        "5:5 가르마",
        "웨이브 레이어드 컷"
    ]

    print("=" * 60)
    print("스타일명 전처리 테스트")
    print("=" * 60)

    for original in test_cases:
        normalized = normalize_style_name(original)
        print(f'"{original}" → "{normalized}"')

    print("=" * 60)
