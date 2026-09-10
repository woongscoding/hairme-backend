"""
core/logging.py 설정 테스트

Lambda 파이썬 런타임은 root 로거에 핸들러를 미리 설치한다. 이때
logging.basicConfig() 는 no-op 이 되어 root 레벨이 WARNING 으로 남고
logger.info(...) 가 CloudWatch 에 전혀 남지 않는다. 아래 테스트는
setup_logging() 이 그 상황에서도 레벨/포맷을 실제로 적용하고, 핸들러를
중복 추가하지 않는지(=같은 로그가 두 줄로 찍히지 않는지) 검증한다.
"""

import contextlib
import json
import logging

import pytest

from core.logging import NOISY_LOGGERS, log_structured, setup_logging


class _CountingHandler(logging.Handler):
    """Lambda 가 미리 설치해둔 root 핸들러를 흉내낸다"""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _is_pytest_handler(handler: logging.Handler) -> bool:
    """pytest logging 플러그인이 매 단계마다 붙였다 떼는 캡처 핸들러"""
    return type(handler).__name__ in ("LogCaptureHandler", "_LiveLoggingNullHandler")


def _app_handlers(root: logging.Logger) -> list:
    return [h for h in root.handlers if not _is_pytest_handler(h)]


@contextlib.contextmanager
def _lambda_like_root(root: logging.Logger):
    """
    pytest 캡처 핸들러를 잠시 떼어 root 를 실제 런타임과 동일한 상태로 만든다.
    (캡처 핸들러가 붙어 있으면 "핸들러 없음" 분기를 검증할 수 없다)
    """
    pytest_handlers = [h for h in root.handlers if _is_pytest_handler(h)]
    for handler in pytest_handlers:
        root.removeHandler(handler)
    try:
        yield
    finally:
        for handler in pytest_handlers:
            root.addHandler(handler)


@pytest.fixture
def clean_root_logger(monkeypatch):
    """root 로거 상태(레벨/애플리케이션 핸들러)를 저장했다가 테스트 후 복원"""
    from config.settings import settings

    monkeypatch.setattr(settings, "LOG_LEVEL", "INFO", raising=False)

    root = logging.getLogger()
    saved_handlers = _app_handlers(root)
    saved_level = root.level

    for handler in saved_handlers:
        root.removeHandler(handler)
    try:
        yield root
    finally:
        for handler in _app_handlers(root):
            root.removeHandler(handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)


def test_sets_root_level_with_preinstalled_handler(clean_root_logger):
    """Lambda 시뮬레이션: 핸들러가 이미 있어도 root 레벨이 INFO 가 되어야 한다"""
    lambda_handler = _CountingHandler()
    lambda_handler.setLevel(logging.WARNING)  # 기본값보다 보수적인 상황
    # Lambda 런타임 핸들러는 자체 포맷터("[LEVEL] 시각 요청ID 메시지" + 줄바꿈 처리)를
    # 가진다. 이걸 교체하면 CloudWatch 에서 줄이 붙어 나오므로 보존되어야 한다.
    runtime_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    lambda_handler.setFormatter(runtime_formatter)
    clean_root_logger.addHandler(lambda_handler)
    clean_root_logger.setLevel(logging.WARNING)

    with _lambda_like_root(clean_root_logger):
        setup_logging()

        # 기존 핸들러를 재사용해야 한다 (중복 추가 금지)
        assert clean_root_logger.handlers == [lambda_handler]

    assert clean_root_logger.level == logging.INFO
    assert lambda_handler.level <= logging.INFO
    # 런타임 포맷터는 그대로여야 한다
    assert lambda_handler.formatter is runtime_formatter


def test_info_record_emitted_exactly_once(clean_root_logger):
    """INFO 로그가 정확히 한 번만 (중복 없이) 방출되어야 한다"""
    lambda_handler = _CountingHandler()
    clean_root_logger.addHandler(lambda_handler)
    clean_root_logger.setLevel(logging.WARNING)

    with _lambda_like_root(clean_root_logger):
        app_logger = setup_logging()
        app_logger.info("hello from lambda")

    messages = [r.getMessage() for r in lambda_handler.records]
    assert messages.count("hello from lambda") == 1


def test_adds_stream_handler_when_root_has_none(clean_root_logger):
    """로컬(핸들러 없음) 환경에서는 StreamHandler 를 하나 추가한다"""
    with _lambda_like_root(clean_root_logger):
        assert clean_root_logger.handlers == []

        setup_logging()

        assert len(clean_root_logger.handlers) == 1
        assert isinstance(clean_root_logger.handlers[0], logging.StreamHandler)

    assert clean_root_logger.level == logging.INFO


def test_repeated_setup_does_not_duplicate_handlers(clean_root_logger):
    """여러 번 호출해도 핸들러가 늘어나지 않는다 (uvicorn reload 등)"""
    with _lambda_like_root(clean_root_logger):
        setup_logging()
        setup_logging()
        setup_logging()

        assert len(clean_root_logger.handlers) == 1


def test_respects_configured_log_level(clean_root_logger, monkeypatch):
    """settings.LOG_LEVEL 을 그대로 반영한다"""
    from config.settings import settings

    monkeypatch.setattr(settings, "LOG_LEVEL", "debug", raising=False)
    setup_logging()
    assert clean_root_logger.level == logging.DEBUG

    monkeypatch.setattr(settings, "LOG_LEVEL", "not-a-level", raising=False)
    setup_logging()
    assert clean_root_logger.level == logging.INFO


def test_noisy_third_party_loggers_stay_at_warning(clean_root_logger):
    """botocore/urllib3/httpx/PIL 등은 INFO 로 내려가면 안 된다"""
    setup_logging()

    for name in ("botocore", "urllib3", "httpx", "PIL"):
        assert name in NOISY_LOGGERS
        assert logging.getLogger(name).level == logging.WARNING


def test_log_structured_emits_single_json_line(clean_root_logger):
    """log_structured 는 JSON 한 줄을 INFO 로 남긴다"""
    handler = _CountingHandler()
    clean_root_logger.addHandler(handler)

    with _lambda_like_root(clean_root_logger):
        setup_logging()
        log_structured("unit_test_event", {"foo": "bar"})

    payloads = [
        json.loads(r.getMessage())
        for r in handler.records
        if r.getMessage().startswith("{")
    ]
    matching = [p for p in payloads if p.get("event_type") == "unit_test_event"]
    assert len(matching) == 1
    assert matching[0]["foo"] == "bar"
    assert matching[0]["timestamp"].endswith("Z")
