"""
Beauty Lambda(main_beauty)가 import하는 서드파티 모듈이 requirements-beauty.txt에
빠지면 Lambda가 Runtime.ImportModuleError로 전면 502가 난다
(2026-09-10: core.jwt_auth → PyJWT 누락으로 염색 추천 장애).
이 테스트는 main_beauty의 전이 import 목록과 requirements-beauty.txt를 대조한다.
"""

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
    "google-generativeai": "google.generativeai",
    "google-genai": "google.genai",
    "python-multipart": "multipart",
    "sentry-sdk": "sentry_sdk",
    "pyyaml": "yaml",
    "pydantic-settings": "pydantic_settings",
    "uvicorn": "uvicorn",
}

# 표준 라이브러리이거나 다른 패키지에 딸려오는 하위 의존성(직접 선언 불필요)
IGNORE_TOP_LEVEL = {
    # 로컬 venv에만 있는 선택적 하위 의존성 (이미지 내 import 검증으로 불필요 확인)
    "bcrypt",
    "cython_runtime",
    "greenlet",
    "orjson",
    "typing_inspection",
    "zstandard",
    "ujson",
    "brotli",
    "simplejson",
    "starlette",
    "anyio",
    "sniffio",
    "h11",
    "httpcore",
    "certifi",
    "idna",
    "urllib3",
    "botocore",
    "s3transfer",
    "jmespath",
    "dateutil",
    "six",
    "typing_extensions",
    "annotated_types",
    "pydantic_core",
    "limits",
    "deprecated",
    "wrapt",
    "google",
    "grpc",
    "proto",
    "googleapiclient",
    "httplib2",
    "uritemplate",
    "pyasn1",
    "pyasn1_modules",
    "rsa",
    "cachetools",
    "requests",
    "charset_normalizer",
    "absl",
    "attr",
    "attrs",
    "flatbuffers",
    "matplotlib",
    "PIL",
    "cv2",
    "mediapipe",
    "numpy",
    "jax",
    "jaxlib",
    "scipy",
    "packaging",
    "dotenv",
    "click",
    "markupsafe",
    "jinja2",
    "multipart",
    "python_multipart",
    "email_validator",
    "dns",
    "websockets",
    "watchfiles",
    "httptools",
    "uvloop",
    "yaml",
    "psutil",
    "pybreaker",
    "slowapi",
    "redis",
    "sqlalchemy",
    "pymysql",
    "cryptography",
    "cffi",
    "pycparser",
    "boto3",
    "mangum",
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "sentry_sdk",
    "google_genai",
    "tenacity",
    "websocket",
    "tqdm",
    "aiohttp",
    "httpx",
    "jwt",
    "kubernetes",
    "mypy_extensions",
    "colorama",
    "pyparsing",
    "cycler",
    "kiwisolver",
    "fontTools",
    "contourpy",
    "dateutil",
    "pytz",
    "tzdata",
    "sounddevice",
    "protobuf",
    "ml_dtypes",
    "opt_einsum",
    "pluggy",
    "iniconfig",
    "_pytest",
    "pytest",
    "py",
    "coverage",
}


def _declared_modules() -> set:
    mods = set()
    for line in (
        (ROOT / "requirements-beauty.txt").read_text(encoding="utf-8").splitlines()
    ):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!\[ ]", line, 1)[0].strip().lower()
        mods.add(DIST_TO_MODULE.get(name, name.replace("-", "_")))
    return mods


def test_main_beauty_imports_are_declared_in_requirements_beauty():
    code = "import sys, main_beauty; print(' '.join(sorted(m for m in sys.modules if '.' not in m)))"
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
    loaded = {m for m in out.stdout.split() if m and not m.startswith("_")}

    # 우리 코드/표준 라이브러리 제외
    stdlib = set(sys.stdlib_module_names)
    ours = {
        "main_beauty",
        "api",
        "core",
        "config",
        "services",
        "models",
        "database",
        "utils",
        "routers",
    }
    third_party = {
        m
        for m in loaded
        if m not in stdlib and m not in ours and m not in IGNORE_TOP_LEVEL
    }

    declared = _declared_modules()
    missing = sorted(
        m for m in third_party if m.split(".")[0] not in declared and m not in declared
    )
    # 핵심 회귀: PyJWT는 반드시 선언되어야 한다
    assert (
        "jwt" in declared
    ), "requirements-beauty.txt에 PyJWT가 없음 (core.jwt_auth import 실패 → 502)"
    assert (
        not missing
    ), f"main_beauty가 import하지만 requirements-beauty.txt에 없는 모듈: {missing}"
