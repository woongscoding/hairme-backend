"""
Monitoring and Observability Configuration

Integrates Sentry for error tracking and performance monitoring.
"""

import logging
import os
from typing import Any, Dict, Optional
from config.settings import settings

logger = logging.getLogger(__name__)


def init_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1,
) -> bool:
    """
    Initialize Sentry error tracking and performance monitoring

    Args:
        dsn: Sentry DSN (from env var SENTRY_DSN if not provided)
        environment: Environment name (production, staging, development)
        traces_sample_rate: Percentage of transactions to trace (0.0-1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0-1.0)

    Returns:
        True if Sentry initialized successfully, False otherwise

    Environment Variables:
        SENTRY_DSN: Sentry project DSN
        SENTRY_ENVIRONMENT: Environment name (overrides environment param)
        SENTRY_TRACES_SAMPLE_RATE: Traces sample rate (overrides param)
        SENTRY_PROFILES_SAMPLE_RATE: Profiles sample rate (overrides param)
    """
    try:
        import sentry_sdk
        from sentry_sdk.integrations.fastapi import FastApiIntegration
        from sentry_sdk.integrations.logging import LoggingIntegration

        # Get configuration from environment or parameters
        sentry_dsn = dsn or os.getenv("SENTRY_DSN")

        if not sentry_dsn:
            logger.warning("⚠️ SENTRY_DSN not configured - Sentry disabled")
            return False

        # Get environment
        sentry_env = (
            os.getenv("SENTRY_ENVIRONMENT") or environment or settings.ENVIRONMENT
        )

        # Get sample rates from env or use defaults
        traces_rate = float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", traces_sample_rate))
        profiles_rate = float(
            os.getenv("SENTRY_PROFILES_SAMPLE_RATE", profiles_sample_rate)
        )

        # Configure Sentry
        sentry_sdk.init(
            dsn=sentry_dsn,
            environment=sentry_env,
            release=f"hairme-backend@{settings.APP_VERSION}",
            # Performance Monitoring
            traces_sample_rate=traces_rate,
            profiles_sample_rate=profiles_rate,
            # Integrations
            integrations=[
                # FastAPI integration
                FastApiIntegration(
                    transaction_style="endpoint",  # Group by endpoint
                    failed_request_status_codes=[500, 501, 502, 503, 504, 505],
                ),
                # Logging integration
                LoggingIntegration(
                    level=logging.INFO,  # Capture INFO and above
                    event_level=logging.ERROR,  # Send ERROR and above to Sentry
                ),
            ],
            # Error filtering
            before_send=before_send_filter,
            # Additional options
            attach_stacktrace=True,
            send_default_pii=False,  # Don't send PII by default
            max_breadcrumbs=50,
            debug=settings.DEBUG,
        )

        logger.info(
            f"✅ Sentry initialized\n"
            f"   Environment: {sentry_env}\n"
            f"   Release: hairme-backend@{settings.APP_VERSION}\n"
            f"   Traces: {traces_rate * 100}%\n"
            f"   Profiles: {profiles_rate * 100}%"
        )

        return True

    except ImportError:
        logger.warning("⚠️ sentry-sdk not installed - Sentry disabled")
        return False

    except Exception as e:
        logger.error(f"❌ Failed to initialize Sentry: {str(e)}")
        return False


def before_send_filter(
    event: Dict[str, Any], hint: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Filter events before sending to Sentry

    Args:
        event: Sentry event dict
        hint: Additional context

    Returns:
        Modified event or None to drop the event
    """
    # Don't send health check errors
    if event.get("transaction") == "GET /api/health":
        return None

    # Don't send expected user errors (4xx)
    if "exception" in event:
        exc_type = event["exception"]["values"][0].get("type", "")

        # Filter out expected exceptions
        expected_exceptions = [
            "NoFaceDetectedException",
            "MultipleFacesException",
            "InvalidFileFormatException",
        ]

        if exc_type in expected_exceptions:
            return None

    # Filter out rate limit test requests
    request = event.get("request", {})
    if request.get("headers", {}).get("X-Test-Request") == "true":
        return None

    return event
