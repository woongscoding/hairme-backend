"""Structured logging utilities for HairMe Backend"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from config.settings import settings

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# CloudWatch 를 도배하는 서드파티 로거는 WARNING 으로 고정한다.
# (root 를 INFO 로 내리면 boto3/urllib3 의 요청 단위 로그까지 전부 올라온다)
NOISY_LOGGERS = (
    "boto3",
    "botocore",
    "s3transfer",
    "urllib3",
    "httpx",
    "httpcore",
    "PIL",
    "matplotlib",
    "asyncio",
)


def _resolve_level() -> int:
    """settings.LOG_LEVEL 을 logging 레벨 정수로 변환 (알 수 없으면 INFO)"""
    level = getattr(logging, str(settings.LOG_LEVEL).upper(), logging.INFO)
    return level if isinstance(level, int) else logging.INFO


def setup_logging() -> logging.Logger:
    """
    애플리케이션 로깅 설정.

    logging.basicConfig() 는 root 로거에 핸들러가 이미 있으면 아무것도 하지
    않는다(no-op). AWS Lambda 파이썬 런타임은 부팅 시점에 root 로거에 자체
    핸들러를 미리 설치하므로, basicConfig 를 쓰면 root 레벨이 WARNING 으로
    남고 logger.info(...) 가 CloudWatch 에 전혀 남지 않는다.

    그래서 여기서는 root 로거를 직접 설정한다:
      - root 레벨을 settings.LOG_LEVEL 로 지정 (Lambda 에서도 INFO 반영)
      - 핸들러가 이미 있으면(=Lambda) 레벨만 맞추고 포맷터/핸들러는 건드리지 않는다
        (추가하면 같은 줄이 두 번 출력되고, 포맷터를 바꾸면 줄바꿈·요청ID가 깨진다)
      - 핸들러가 없으면(=로컬/uvicorn) StreamHandler 를 하나 추가한다

    여러 번 호출해도 핸들러가 늘어나지 않는다(idempotent).
    """
    level = _resolve_level()
    formatter = logging.Formatter(LOG_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if root_logger.handlers:
        # Lambda(또는 이미 로깅이 구성된 환경): 핸들러를 추가하지 않는다(중복 출력 방지).
        # 포맷터도 건드리지 않는다 - Lambda 런타임 핸들러의 포맷터는
        # "[LEVEL]	시각	요청ID	메시지" 형식과 줄바꿈 처리를 담당하므로,
        # 교체하면 CloudWatch 에서 줄이 붙어 나오고 요청 ID 가 사라진다(2026-09-10 확인).
        for handler in root_logger.handlers:
            # 핸들러가 root 보다 엄격하면 레벨만 완화한다.
            # (반대로 더 관대한 핸들러의 레벨을 올리지는 않는다 - pytest caplog 등)
            if handler.level > level:
                handler.setLevel(level)
    else:
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    # 서드파티 소음 억제
    for noisy in NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


logger = setup_logging()


def log_structured(event_type: str, data: Dict[str, Any]) -> None:
    """
    Log structured JSON data for CloudWatch Logs Insights analysis

    Args:
        event_type: Type of event (e.g., "analysis_start", "cache_hit")
        data: Dictionary containing event data
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        **data,
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
