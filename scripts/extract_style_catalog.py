# -*- coding: utf-8 -*-
"""헤어스타일 카탈로그 추출 스크립트

data_source/style_embeddings.npz(ML 스타일 310개) + hairstyle_gender.json(성별)
+ trending_hairstyles.json(트렌드 스타일)에서 이미지 생성용 스타일 카탈로그를 만든다.

원본 스타일명에는 표기 변형이 많아("리젠트 컷", "리젠트컷 (Regent Cut)", ...)
정규화된 대표명(canonical)으로 그룹핑하고, 시각적으로 동일한 스타일은
ALIAS_MAP으로 하나의 image_key에 묶는다. 이미지 생성은 image_key 단위로 1회만 수행.

출력: data_source/style_images/styles.json
- ml_styles: hairstyle_id(npz 인덱스) → canonical/image_key 매핑 (API 서빙 시 사용)
- trending_styles: 트렌드 스타일명 → image_key 매핑
- image_keys: 이미지를 생성해야 하는 최종 목록 (성별 포함)

새 스타일이 추가되면 이 스크립트를 재실행한 뒤, prompts.json에 누락된
image_key의 프롬프트를 추가하면 된다 (누락 목록을 마지막에 출력).

사용법:
    python scripts/extract_style_catalog.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data_source"
OUT_DIR = DATA_DIR / "style_images"

EMBEDDINGS_PATH = DATA_DIR / "style_embeddings.npz"
GENDER_PATH = DATA_DIR / "hairstyle_gender.json"
TRENDING_PATH = DATA_DIR / "trending_hairstyles.json"
PROMPTS_PATH = OUT_DIR / "prompts.json"
STYLES_OUT_PATH = OUT_DIR / "styles.json"

# 시각적으로 동일한 스타일을 하나의 이미지로 묶는 별칭 매핑 (canonical → image_key)
ALIAS_MAP = {
    "포마드스타일": "포마드",
    "올백스타일": "올백",
    "올백포마드": "올백",
    "바버스타일페이드컷": "바버스타일",
    "볼륨매직스타일": "볼륨매직",
    "스킨페이드컷": "스킨페이드",
    "장발파마": "장발펌",
    "장발펌스타일": "장발펌",
    "장발히피스타일": "장발히피펌",
    "시스루댄디컷": "시스루댄디",
    "시스루뱅댄디컷": "시스루댄디",
    "시스루뱅스타일": "시스루뱅",
    "시스루뱅앞머리스타일": "시스루뱅",
    "시스루뱅다운펌": "시스루뱅",
    "시스루뱅앞머리+쉐도우펌": "시스루뱅",
    "짧은크롭컷": "크롭컷",
    "크롭컷+다운펌": "크롭컷",
    "짧은스포츠스타일": "스왓컷",
    "짧은스포츠머리": "스왓컷",
    "5:5가르마": "가르마스타일",
    "5:5가르마펌": "가르마펌",
    "소프트투블럭컷": "소프트투블럭",
    "소프트투블럭댄디컷": "소프트투블럭",
    "강한투블럭": "투블럭",
    "숏리젠트컷": "리젠트컷",
}


def canonical_name(raw: str) -> str:
    """원본 스타일명 → 대표명

    1) 괄호 안 내용 제거 (영문 표기, 세부 변형 설명)
    2) ": " 뒤의 설명문 제거 ("5:5"처럼 공백 없는 콜론은 보존)
    3) 띄어쓰기 제거 (utils.style_preprocessor.normalize_style_name과 동일 규칙)
    """
    name = re.sub(r"\([^)]*\)", "", raw)
    name = name.split(": ")[0]
    name = name.strip().rstrip(":").strip()
    return name.replace(" ", "")


def resolve_image_key(canonical: str) -> str:
    return ALIAS_MAP.get(canonical, canonical)


def vote_gender(genders: list) -> str:
    """변형들의 성별 라벨을 하나로 합침: male/female 혼재 시 unisex, 아니면 다수결"""
    counts = Counter(genders)
    if counts.get("male") and counts.get("female"):
        return "unisex"
    if counts.get("unisex", 0) >= max(counts.get("male", 0), counts.get("female", 0)):
        return "unisex"
    return counts.most_common(1)[0][0]


def main():
    # Windows 콘솔(cp949)에서 한글/이모지 출력 오류 방지
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 1. ML 스타일 (npz 인덱스 = hairstyle_id)
    data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
    raw_styles = [str(s) for s in data["styles"].tolist()]

    with open(GENDER_PATH, encoding="utf-8") as f:
        gender_map = json.load(f)  # {원본 스타일명: "male"|"female"|"unisex"}

    ml_styles = []
    key_genders = defaultdict(list)
    for idx, raw in enumerate(raw_styles):
        canon = canonical_name(raw)
        key = resolve_image_key(canon)
        gender = gender_map.get(raw, "unisex")
        ml_styles.append(
            {
                "hairstyle_id": idx,
                "raw_name": raw,
                "canonical": canon,
                "image_key": key,
                "gender": gender,
            }
        )
        key_genders[key].append(gender)

    # 2. 트렌드 스타일 (hairstyle_id 없음, 그룹 라벨이 성별)
    with open(TRENDING_PATH, encoding="utf-8") as f:
        trending = json.load(f)

    trending_styles = []
    for group_gender, names in trending.items():
        for name in names:
            canon = canonical_name(name)
            key = resolve_image_key(canon)
            trending_styles.append(
                {
                    "style_name": name,
                    "canonical": canon,
                    "image_key": key,
                    "gender": group_gender,
                }
            )
            key_genders[key].append(group_gender)

    # 3. image_key별 성별 확정 → 생성할 (성별, image_key) 목록
    image_keys = []
    for key in sorted(key_genders):
        gender = vote_gender(key_genders[key])
        # unisex는 남/녀 버전을 각각 생성
        target_genders = ["male", "female"] if gender == "unisex" else [gender]
        image_keys.append(
            {
                "image_key": key,
                "gender": gender,
                "generate_for": target_genders,
                "variant_count": len(key_genders[key]),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "source": {
            "ml_styles": len(ml_styles),
            "trending_styles": len(trending_styles),
            "image_keys": len(image_keys),
        },
        "image_keys": image_keys,
        "ml_styles": ml_styles,
        "trending_styles": trending_styles,
    }
    with open(STYLES_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"styles.json 저장: {STYLES_OUT_PATH}")
    print(
        f"ML {len(ml_styles)}개 + 트렌드 {len(trending_styles)}개 "
        f"→ image_key {len(image_keys)}개"
    )
    total_images = sum(len(k["generate_for"]) for k in image_keys) * 2
    print(f"전체 생성 시 이미지 수: {total_images}장 (스타일·성별당 2장)")

    # 4. prompts.json 커버리지 검증 (스타일 추가 시 누락 확인용)
    if PROMPTS_PATH.exists():
        with open(PROMPTS_PATH, encoding="utf-8") as f:
            prompts = json.load(f)
        covered = {p["image_key"] for p in prompts["styles"]}
        missing = [k["image_key"] for k in image_keys if k["image_key"] not in covered]
        if missing:
            print(f"\n⚠️ prompts.json에 프롬프트가 없는 image_key {len(missing)}개:")
            for key in missing:
                print(f"  - {key}")
            sys.exit(1)
        print("✅ 모든 image_key에 프롬프트가 존재합니다.")
    else:
        print(f"\n⚠️ {PROMPTS_PATH} 없음 - 프롬프트 세트를 먼저 작성하세요.")


if __name__ == "__main__":
    main()
