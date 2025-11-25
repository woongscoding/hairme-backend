"""
헤어스타일 성별 메타데이터 자동 생성

310개 헤어스타일을 분석하여 각 스타일의 성별 태그를 자동으로 분류합니다.
- male: 남성용 헤어스타일
- female: 여성용 헤어스타일
- unisex: 남녀 공용 헤어스타일
"""
import numpy as np
import json

# 임베딩 파일 로드
data = np.load('data_source/style_embeddings.npz', allow_pickle=False)
styles = data['styles'].tolist()

# 성별 분류 키워드
MALE_KEYWORDS = [
    '남성', '남자', '모히칸', '크루컷', '투블럭', '언더컷', '포마드',
    '스포츠', '군인', '남학생', '페이드', 'fade', 'undercut',
    '크롭', 'crop', '남자 스포츠', '짧은 남자', '매쉬업', '매시업'
]

FEMALE_KEYWORDS = [
    '여성', '여자', '긴 머리', '웨이브', '펌', '여학생', '포니테일',
    '생머리', '롱헤어', 'long hair', '여자 스포츠', '긴 생머리',
    '긴 웨이브', '부드러운', '우아한', '여성스러운', '긴 스트레이트'
]

UNISEX_KEYWORDS = [
    '가르마', '단발', '보브', 'bob', '숏컷', '중단발', '쉼표머리',
    '센터 파팅', '쿼터', '애쉬', '실버', '컬러', '염색', '중단발머리'
]

# 자동 분류 함수
def classify_gender(style_name: str) -> str:
    """
    헤어스타일명으로 성별 분류

    Returns:
        "male", "female", "unisex"
    """
    style_lower = style_name.lower()

    # 1. 명시적 키워드 매칭
    male_match = any(keyword in style_lower for keyword in MALE_KEYWORDS)
    female_match = any(keyword in style_lower for keyword in FEMALE_KEYWORDS)
    unisex_match = any(keyword in style_lower for keyword in UNISEX_KEYWORDS)

    # 2. 우선순위 결정
    if male_match and not female_match:
        return "male"
    elif female_match and not male_match:
        return "female"
    elif unisex_match:
        return "unisex"
    else:
        # 3. 휴리스틱 규칙
        # "짧은" + "컷" = 남녀 공용 (단발)
        if ("짧은" in style_lower or "short" in style_lower) and ("컷" in style_lower or "cut" in style_lower):
            return "unisex"

        # "긴" = 여성용 (긴 머리는 주로 여성)
        if "긴" in style_lower or "long" in style_lower:
            return "female"

        # 기본값: 남녀 공용
        return "unisex"


# 자동 분류 실행
gender_metadata = {}
male_count = 0
female_count = 0
unisex_count = 0

print("[INFO] 헤어스타일 성별 분류 시작...")
print(f"[INFO] 총 {len(styles)}개 스타일 분석 중...\n")

for style in styles:
    gender = classify_gender(style)
    gender_metadata[style] = gender

    if gender == "male":
        male_count += 1
    elif gender == "female":
        female_count += 1
    else:
        unisex_count += 1

# 결과 저장
output_file = 'data_source/hairstyle_gender.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(gender_metadata, f, ensure_ascii=False, indent=2)

print(f"[SUCCESS] 성별 메타데이터 생성 완료!")
print(f"[INFO] 저장 위치: {output_file}")
print(f"\n[STATS] 분류 결과:")
print(f"  - 남성용: {male_count}개 ({male_count/len(styles)*100:.1f}%)")
print(f"  - 여성용: {female_count}개 ({female_count/len(styles)*100:.1f}%)")
print(f"  - 남녀 공용: {unisex_count}개 ({unisex_count/len(styles)*100:.1f}%)")

# 샘플 출력
print(f"\n[SAMPLE] 남성용 스타일 (처음 10개):")
male_samples = [s for s, g in gender_metadata.items() if g == "male"][:10]
for i, style in enumerate(male_samples, 1):
    print(f"  {i}. {style}")

print(f"\n[SAMPLE] 여성용 스타일 (처음 10개):")
female_samples = [s for s, g in gender_metadata.items() if g == "female"][:10]
for i, style in enumerate(female_samples, 1):
    print(f"  {i}. {style}")

print(f"\n[SAMPLE] 남녀 공용 스타일 (처음 10개):")
unisex_samples = [s for s, g in gender_metadata.items() if g == "unisex"][:10]
for i, style in enumerate(unisex_samples, 1):
    print(f"  {i}. {style}")
