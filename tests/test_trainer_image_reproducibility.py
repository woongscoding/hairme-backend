"""
Trainer 이미지 재현성 회귀 테스트

대상: lambda_trainer/Dockerfile, lambda_trainer/requirements.txt

requirements.txt 가 트레이너 이미지 의존성의 단일 소스여야 한다.
- Dockerfile 은 requirements.txt 만 설치하고 버전을 다시 적지 않는다
  (두 곳에 적으면 한쪽만 고쳐져 어긋난다 - 과거 requirements.txt 는 torch>=2.0.0
  범위라 Dockerfile 핀과 달랐다).
- torch 는 CPU 전용 휠로 정확히 고정한다. PyPI 의 Linux torch 휠은 CUDA 빌드라
  Lambda 이미지가 수 GB 로 커진다.
"""

import io
import re

REQUIREMENTS = "lambda_trainer/requirements.txt"
DOCKERFILE = "lambda_trainer/Dockerfile"


def _read(path):
    with io.open(path, encoding="utf-8") as f:
        return f.read()


def _requirement_lines():
    """주석/빈 줄을 뺀 실제 요구사항 줄"""
    lines = []
    for raw in _read(REQUIREMENTS).splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def test_dockerfile_installs_from_requirements_file():
    """Dockerfile 은 requirements.txt 를 복사해 그대로 설치한다"""
    dockerfile = _read(DOCKERFILE)

    assert re.search(r"^COPY\s+requirements\.txt\b", dockerfile, re.MULTILINE)
    assert re.search(r"pip install[^\n]*-r[^\n]*requirements\.txt", dockerfile)


def test_dockerfile_does_not_repin_packages():
    """Dockerfile 에 패키지 버전을 다시 적지 않는다 (단일 소스 유지)"""
    dockerfile = _read(DOCKERFILE)

    # RUN 명령만 검사 (주석의 설명 문구는 제외)
    run_lines = [
        line
        for line in dockerfile.splitlines()
        if line.strip().startswith("RUN") or line.strip().startswith("pip")
    ]
    joined = "\n".join(run_lines)

    for package in ("torch", "numpy", "boto3"):
        assert package not in joined, (
            f"Dockerfile RUN 에서 {package} 를 직접 설치하면 requirements.txt 와 "
            "어긋날 수 있다"
        )


def test_torch_pinned_to_exact_cpu_build():
    """torch 는 2.5.1 CPU 휠로 정확히 고정한다"""
    lines = _requirement_lines()

    torch_lines = [line for line in lines if line.startswith("torch")]
    assert torch_lines == ["torch==2.5.1+cpu"]


def test_pytorch_cpu_index_is_configured():
    """+cpu 휠은 PyTorch CPU 인덱스에만 있으므로 인덱스가 반드시 선언돼야 한다"""
    lines = _requirement_lines()

    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lines


def test_numpy_and_boto3_bounded():
    """numpy 1.x 고정 (2.x 금지), boto3 는 상한 포함 범위"""
    lines = _requirement_lines()

    assert "numpy>=1.26,<2.0" in lines
    assert "boto3>=1.28,<2.0" in lines


def test_no_unbounded_requirements():
    """상한 없는 >= 범위는 빌드 재현성을 깨므로 금지"""
    for line in _requirement_lines():
        if line.startswith("--"):
            continue
        if ">=" in line:
            assert "<" in line, f"상한 없는 범위: {line}"
