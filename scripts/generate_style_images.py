# -*- coding: utf-8 -*-
"""헤어스타일 예시 이미지 배치 생성 스크립트 (Gemini 3 Pro Image)

data_source/style_images/prompts.json의 프롬프트 세트를 읽어
스타일·성별당 2장씩 예시 이미지를 생성한다.

- 모델: prompts.json의 model 필드 (기본 gemini-3-pro-image, 장당 약 $0.13)
  * 원래 imagen-4.0-ultra-generate-001 예정이었으나 Imagen 4 계열이
    신규 프로젝트에 차단되어(404) Gemini 3 Pro Image로 대체함.
  * imagen-* 모델명을 지정하면 generate_images API로 자동 분기한다.
- API 키: GEMINI_API_KEY 환경변수 (없으면 프로젝트 루트 .env에서 로드)
- 출력: data_source/style_images/generated/{gender}_{id:03d}_{image_key}_{n}.png|jpg
- 이미 존재하는 파일은 건너뛰므로 중단 후 재실행해도 안전하다.

사용법:
    # 1) 테스트 생성 (남/녀 각 2개 스타일, 8장) - 품질 검수용
    python scripts/generate_style_images.py --test

    # 2) 검수 통과 후 전량 생성
    python scripts/generate_style_images.py --all --yes

    # 3) 프롬프트 수정 후 특정 스타일만 재생성 (기존 파일 삭제 후 실행)
    python scripts/generate_style_images.py --only 크롭컷,보브컷

    # 프롬프트만 확인 (API 호출 없음)
    python scripts/generate_style_images.py --all --dry-run
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_PATH = PROJECT_ROOT / "data_source" / "style_images" / "prompts.json"
OUTPUT_DIR = PROJECT_ROOT / "data_source" / "style_images" / "generated"

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds (지수 백오프 기준값)
COST_PER_IMAGE = 0.13  # USD, Gemini 3 Pro Image 대략치 (Imagen 4 Ultra는 0.06)
CONFIRM_THRESHOLD = 10  # 이 장수 초과 생성 시 --yes 필요


def load_api_key() -> str:
    """GEMINI_API_KEY 환경변수 로드 (없으면 .env 시도). 키 값은 절대 출력하지 않는다."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        env_path = PROJECT_ROOT / ".env"
        if env_path.exists():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_path)
                key = os.environ.get("GEMINI_API_KEY")
            except ImportError:
                pass
    if not key:
        print("❌ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        sys.exit(1)
    return key


def load_prompts() -> dict:
    import json

    with open(PROMPTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def build_work_items(data: dict, args) -> list:
    """생성 대상 (entry, n) 목록 구성. 이미 존재하는 파일은 제외."""
    entries = data["styles"]

    if args.test:
        test_set = set(data.get("test_set", []))
        entries = [e for e in entries if f"{e['gender']}/{e['image_key']}" in test_set]
    elif args.only:
        only_keys = {k.strip() for k in args.only.split(",") if k.strip()}
        entries = [e for e in entries if e["image_key"] in only_keys]
        found = {e["image_key"] for e in entries}
        for missing in only_keys - found:
            print(f"⚠️ prompts.json에 없는 image_key: {missing}")

    images_per_style = data.get("images_per_style", 2)
    template = data["template"]
    gender_words = data["gender_words"]

    items = []
    skipped = 0
    for entry in entries:
        prompt = template.format(
            gender_word=gender_words[entry["gender"]],
            hair_description=entry["hair_description"],
        )
        for n in range(1, images_per_style + 1):
            stem = f"{entry['gender']}_{entry['id']:03d}_{entry['image_key']}_{n}"
            # 확장자는 응답 포맷에 따라 달라지므로(png/jpg) 스템 기준으로 존재 확인
            if any(OUTPUT_DIR.glob(f"{stem}.*")):
                skipped += 1
                continue
            items.append({"entry": entry, "n": n, "prompt": prompt, "stem": stem})

    if skipped:
        print(f"⏭️ 이미 존재하는 파일 {skipped}장 건너뜀")
    return items


_MIME_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


def _request_image_bytes(client, model: str, aspect_ratio: str, prompt: str):
    """모델 종류에 따라 이미지 1장 생성. (bytes, mime_type) 반환, 미반환 시 (None, None)."""
    from google.genai import types

    if model.startswith("imagen"):
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                aspect_ratio=aspect_ratio,
                person_generation="ALLOW_ADULT",
                output_mime_type="image/png",
            ),
        )
        images = getattr(response, "generated_images", None)
        if not images:
            return None, None
        return images[0].image.image_bytes, "image/png"

    # Gemini 이미지 모델 (gemini-3-pro-image 등)은 generate_content 사용
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
        ),
    )
    for candidate in response.candidates or []:
        for part in candidate.content.parts or []:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, inline.mime_type
    return None, None


def generate_one(client, model: str, aspect_ratio: str, item: dict) -> bool:
    """이미지 1장 생성 (재시도 포함). 성공 여부 반환."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            data, mime = _request_image_bytes(
                client, model, aspect_ratio, item["prompt"]
            )
            if data is None:
                # 세이프티 필터 등으로 이미지가 반환되지 않은 경우
                print(
                    f"   ⚠️ 이미지 미반환 (시도 {attempt}/{MAX_RETRIES}) - 프롬프트 확인 필요"
                )
                time.sleep(RETRY_DELAY * attempt)
                continue

            ext = _MIME_EXT.get(mime, ".png")
            out_path = OUTPUT_DIR / f"{item['stem']}{ext}"
            out_path.write_bytes(data)
            item["saved_path"] = out_path
            return True
        except Exception as e:
            # 에러 메시지에 키가 섞이지 않도록 타입과 요약만 출력
            print(
                f"   ⚠️ 실패 (시도 {attempt}/{MAX_RETRIES}): {type(e).__name__}: {str(e)[:200]}"
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    return False


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="헤어스타일 예시 이미지 배치 생성")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test", action="store_true", help="테스트 세트만 생성 (검수용)"
    )
    group.add_argument("--all", action="store_true", help="전체 스타일 생성")
    group.add_argument("--only", type=str, help="특정 image_key만 생성 (쉼표 구분)")
    parser.add_argument(
        "--dry-run", action="store_true", help="프롬프트만 출력, API 호출 없음"
    )
    parser.add_argument("--yes", action="store_true", help="대량 생성 확인 생략")
    args = parser.parse_args()

    data = load_prompts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    items = build_work_items(data, args)
    if not items:
        print("생성할 이미지가 없습니다 (모두 존재하거나 대상 없음).")
        return

    cost = len(items) * COST_PER_IMAGE
    print(f"생성 대상: {len(items)}장 (예상 비용 약 ${cost:.2f})")

    if args.dry_run:
        for item in items:
            print(f"\n[{item['stem']}]")
            print(f"  {item['prompt']}")
        return

    if len(items) > CONFIRM_THRESHOLD and not args.yes:
        print(f"❌ {CONFIRM_THRESHOLD}장 초과 생성은 --yes 플래그가 필요합니다.")
        print("   테스트 검수를 먼저 진행했는지 확인하세요 (--test).")
        sys.exit(1)

    api_key = load_api_key()
    from google import genai

    client = genai.Client(api_key=api_key)
    model = data.get("model", "imagen-4.0-ultra-generate-001")
    aspect_ratio = data.get("aspect_ratio", "3:4")

    success, failed = 0, []
    start = time.time()
    for i, item in enumerate(items, 1):
        name = item["stem"]
        print(f"[{i}/{len(items)}] {name} 생성 중...")
        if generate_one(client, model, aspect_ratio, item):
            success += 1
            print(f"   ✅ 저장: {item['saved_path'].relative_to(PROJECT_ROOT)}")
        else:
            failed.append(name)
            print(f"   ❌ 최종 실패: {name}")

    elapsed = time.time() - start
    print(f"\n완료: 성공 {success}장 / 실패 {len(failed)}장 ({elapsed:.0f}초)")
    print(f"실제 비용 약 ${success * COST_PER_IMAGE:.2f}")
    if failed:
        print("실패 목록 (재실행하면 실패분만 다시 시도):")
        for name in failed:
            print(f"  - {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
