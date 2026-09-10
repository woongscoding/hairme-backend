"""
Beauty Lambda(main_beauty)가 직접 import하는 서드파티 모듈이 requirements-beauty.txt에
빠지면 Lambda가 Runtime.ImportModuleError로 전면 502가 난다
(2026-09-10: core.jwt_auth → PyJWT 누락으로 염색 추천 장애).

방식: main_beauty를 서브프로세스에서 import해 실제로 로드된 *우리 소스 파일* 목록을 얻고,
그 파일들의 최상위 import 문을 AST로 파싱해 서드파티 최상위 패키지를 뽑은 뒤
requirements-beauty.txt 선언과 대조한다. (환경에 따라 달라지는 하위 의존성은 검사하지 않음)
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 배포판 이름 → import 이름 (다른 경우만)
DIST_TO_MODULE = {
    "pyjwt": "jwt",
    "pillow": "PIL",
    "opencv-python-headless": "cv2",
    "google-generativeai": "google",
    "google-genai": "google",
    "python-multipart": "multipart",
    "sentry-sdk": "sentry_sdk",
    "pyyaml": "yaml",
    "pydantic-settings": "pydantic_settings",
    "scikit-learn": "sklearn",
}

# 선언된 패키지가 끌고 오는 것이 확실한 모듈 (직접 선언 불필요)
PROVIDED_BY_DECLARED = {
    "starlette": "fastapi",
    "botocore": "boto3",
    "PIL": "pillow",
    "google": "google-genai",
    "absl": "mediapipe",
    "matplotlib": "mediapipe",
    "grpc": "google-generativeai",
    # config/secrets.py 의 requests: google-generativeai(google-api-core) 하위 의존성으로 이미지에 포함됨 (2026-09-10 이미지 확인)
    "requests": "google-generativeai",
}

OUR_TOP_LEVEL = {
    "api",
    "core",
    "config",
    "services",
    "models",
    "database",
    "utils",
    "routers",
    "main",
    "main_beauty",
}


def _declared_modules() -> set:
    mods = set()
    text = (ROOT / "requirements-beauty.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[ ]", line, 1)[0].strip().lower()
        mods.add(DIST_TO_MODULE.get(name, name.replace("-", "_")))
    return mods


def _loaded_source_files() -> list:
    code = "import sys, main_beauty; print(' '.join(sorted(getattr(m, '__file__', '') or '' for m in list(sys.modules.values()))))"
    env = dict(
        os.environ,
        GEMINI_API_KEY="test_key",
        USE_DYNAMODB="false",
        LAMBDA_TYPE="beauty",
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    files = []
    for line in out.stdout.split():
        p = Path(line) if line else None
        if (
            p
            and p.suffix == ".py"
            and ROOT in p.resolve().parents
            and "venv" not in p.parts
            and "tests" not in p.parts
        ):
            files.append(p)
    assert files, "main_beauty 로드 시 우리 소스 파일이 하나도 잡히지 않음"
    return files


def _direct_third_party_imports(files: list) -> dict:
    """{top_level_module: first_file_that_imports_it}"""
    stdlib = set(sys.stdlib_module_names)
    found = {}
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            for n in names:
                top = n.split(".")[0]
                if top in stdlib or top in OUR_TOP_LEVEL or top.startswith("_"):
                    continue
                found.setdefault(top, str(f.relative_to(ROOT)))
    return found


def test_main_beauty_direct_imports_are_declared_in_requirements_beauty():
    declared = _declared_modules()
    assert (
        "jwt" in declared
    ), "requirements-beauty.txt에 PyJWT가 없음 (core.jwt_auth import 실패 → 502)"

    imports = _direct_third_party_imports(_loaded_source_files())
    missing = {
        mod: src
        for mod, src in imports.items()
        if mod not in declared and PROVIDED_BY_DECLARED.get(mod) is None
    }
    # 개발 도구(테스트에서만 쓰는 것)는 제외
    for dev_only in ("pytest", "moto"):
        missing.pop(dev_only, None)
    assert not missing, (
        "main_beauty가 로드하는 소스가 직접 import하지만 requirements-beauty.txt에 없는 패키지: "
        f"{missing}"
    )
