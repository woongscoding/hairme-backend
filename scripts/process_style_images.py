# -*- coding: utf-8 -*-
"""헤어스타일 예시 이미지 후처리 스크립트 (5단계)

data_source/style_images/generated/의 원본 이미지(jpg/png)를
서빙용 WebP로 변환하고(긴 변 1024px, 품질 85) 매니페스트를 생성한다.

- 출력: data_source/style_images/webp/{stem}.webp
- 매니페스트: data_source/style_images/image_manifest.json
  {"male/크롭컷": {"file": "male_072_크롭컷_1.webp", "alts": [...]}, ...}
  기본으로 각 스타일의 1번 이미지를 대표로 선택한다.
  다른 번호를 대표로 쓰려면 --prefer "male/크롭컷=2" 형태로 지정.

WebP와 매니페스트는 레포에 커밋되어 Lambda 이미지에 번들되고,
GET /api/styles/images/{filename} 엔드포인트로 서빙된다.

사용법:
    python scripts/process_style_images.py
    python scripts/process_style_images.py --prefer "male/크롭컷=2,female/보브컷=2"
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
STYLE_DIR = PROJECT_ROOT / "data_source" / "style_images"
GENERATED_DIR = STYLE_DIR / "generated"
WEBP_DIR = STYLE_DIR / "webp"
MANIFEST_PATH = STYLE_DIR / "image_manifest.json"

MAX_SIZE = 1024  # 긴 변 기준 px
WEBP_QUALITY = 85

# 파일명 패턴: {gender}_{id:03d}_{image_key}_{n}.{ext}
FILENAME_RE = re.compile(r"^(male|female)_(\d{3})_(.+)_(\d+)\.(jpg|jpeg|png)$")


def convert_to_webp(src: Path, dst: Path) -> None:
    with Image.open(src) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        img.save(dst, "WEBP", quality=WEBP_QUALITY)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="스타일 이미지 WebP 변환 + 매니페스트 생성"
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default="",
        help='대표 이미지 번호 지정 (예: "male/크롭컷=2,female/보브컷=2")',
    )
    args = parser.parse_args()

    prefer = {}
    for token in args.prefer.split(","):
        token = token.strip()
        if not token:
            continue
        entry_key, _, n = token.partition("=")
        prefer[entry_key.strip()] = int(n)

    if not GENERATED_DIR.exists():
        print(f"❌ 원본 디렉토리 없음: {GENERATED_DIR}")
        sys.exit(1)

    WEBP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. WebP 변환 (기존 파일은 건너뜀)
    converted, skipped = 0, 0
    sources = sorted(p for p in GENERATED_DIR.iterdir() if FILENAME_RE.match(p.name))
    for src in sources:
        dst = WEBP_DIR / (src.stem + ".webp")
        if dst.exists():
            skipped += 1
            continue
        convert_to_webp(src, dst)
        converted += 1
    print(f"WebP 변환: {converted}장 변환, {skipped}장 건너뜀 (총 {len(sources)}장)")

    # 2. 매니페스트 생성: (gender, image_key) → 대표 파일 + 대안 목록
    groups = defaultdict(dict)  # "gender/image_key" -> {n: filename}
    for webp in sorted(WEBP_DIR.glob("*.webp")):
        m = FILENAME_RE.match(webp.stem + ".jpg")  # 패턴 재사용을 위해 확장자 부착
        if not m:
            print(f"⚠️ 파일명 패턴 불일치 (매니페스트 제외): {webp.name}")
            continue
        gender, _sid, image_key, n = m.group(1), m.group(2), m.group(3), int(m.group(4))
        groups[f"{gender}/{image_key}"][n] = webp.name

    manifest = {}
    for entry_key in sorted(groups):
        by_n = groups[entry_key]
        chosen_n = prefer.get(entry_key, 1)
        if chosen_n not in by_n:
            chosen_n = min(by_n)  # 대표 번호가 없으면 가장 낮은 번호 사용
        manifest[entry_key] = {
            "file": by_n[chosen_n],
            "alts": [by_n[n] for n in sorted(by_n) if n != chosen_n],
        }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"매니페스트 저장: {MANIFEST_PATH} ({len(manifest)}개 항목)")

    total_bytes = sum(p.stat().st_size for p in WEBP_DIR.glob("*.webp"))
    print(f"WebP 총 용량: {total_bytes / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    main()
