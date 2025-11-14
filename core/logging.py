"""Structured logging utilities for HairMe Backend"""

import json
import logging
from datetime import datetime
from typing import Dict, Any

from config.settings import settings


def setup_logging() -> logging.Logger:
    """Configure application logging with proper formatting"""
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
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
        **data
    }
    logger.info(json.dumps(log_entry, ensure_ascii=False))
