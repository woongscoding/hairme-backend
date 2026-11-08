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


if __name__ == "__main__":
    # 테스트
    test_cases = [
        "시스루 뱅",
        "레이어드 컷",
        "C컬 단발",
        "단발 보브",
        "  허쉬컷  ",
        "5:5 가르마",
        "웨이브 레이어드 컷",
    ]

    print("=" * 60)
    print("스타일명 전처리 테스트")
    print("=" * 60)

    for original in test_cases:
        normalized = normalize_style_name(original)
        print(f'"{original}" → "{normalized}"')

    print("=" * 60)
