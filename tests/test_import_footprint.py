"""Task 4: `import main` 이 무거운 ML 라이브러리를 끌고 오지 않는지 검증

torch / mediapipe / cv2 / sentence_transformers 를 모듈 import 시점에 로드하면
Lambda init 단계에서 3초 이상이 소모되어 10초 init 타임아웃에 근접한다.
이 테스트는 별도 프로세스에서 `import main` 을 수행해 sys.modules 를 확인한다.
"""

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HEAVY_MODULES = ("torch", "mediapipe", "cv2", "sentence_transformers")

_PROBE = (
    "import sys, main; "
    "print('HEAVY:' + ','.join("
    "m for m in ('torch', 'mediapipe', 'cv2', 'sentence_transformers') "
    "if m in sys.modules))"
)


def _run_probe(module_name: str) -> str:
    env = dict(os.environ)
    env["GEMINI_API_KEY"] = "test_key"
    env["USE_DYNAMODB"] = "false"
    env["TESTING"] = "true"
    env["PYTHONIOENCODING"] = "utf-8"

    probe = _PROBE.replace("import main", f"import {module_name}")

    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(PROJECT_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=180,
    )

    assert result.returncode == 0, (
        f"import {module_name} 실패 (exit={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    lines = [ln for ln in result.stdout.splitlines() if ln.startswith("HEAVY:")]
    assert lines, f"probe 출력에 HEAVY 라인이 없음:\n{result.stdout}"
    return lines[-1][len("HEAVY:") :].strip()


class TestImportFootprint:
    def test_import_main_does_not_load_heavy_modules(self):
        """`import main` 후 torch/mediapipe/cv2/sentence_transformers 가 없어야 한다"""
        loaded = _run_probe("main")
        assert loaded == "", (
            "main import 시 무거운 모듈이 로드됨: "
            f"{loaded} (해당 import 를 함수 안이나 TYPE_CHECKING 으로 옮길 것)"
        )

    def test_import_analyze_endpoint_does_not_load_torch(self):
        """analyze 엔드포인트 모듈 단독 import 도 torch 를 끌어오면 안 된다"""
        loaded = _run_probe("api.endpoints.analyze")
        assert loaded == "", f"api.endpoints.analyze import 시 로드됨: {loaded}"
