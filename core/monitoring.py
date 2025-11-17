"""
Monitoring and Observability Configuration

Integrates Sentry for error tracking and performance monitoring.
"""

import logging
import os
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


def init_sentry(
    dsn: Optional[str] = None,
    environment: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1
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
        sentry_dsn = dsn or os.getenv('SENTRY_DSN')

        if not sentry_dsn:
            logger.warning("⚠️ SENTRY_DSN not configured - Sentry disabled")
            return False

        # Get environment
        sentry_env = os.getenv('SENTRY_ENVIRONMENT') or environment or settings.ENVIRONMENT

        # Get sample rates from env or use defaults
        traces_rate = float(os.getenv('SENTRY_TRACES_SAMPLE_RATE', traces_sample_rate))
        profiles_rate = float(os.getenv('SENTRY_PROFILES_SAMPLE_RATE', profiles_sample_rate))

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
                    failed_request_status_codes=[500, 501, 502, 503, 504, 505]
                ),

                # Logging integration
                LoggingIntegration(
                    level=logging.INFO,  # Capture INFO and above
                    event_level=logging.ERROR  # Send ERROR and above to Sentry
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


def before_send_filter(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter events before sending to Sentry

    Args:
        event: Sentry event dict
        hint: Additional context

    Returns:
        Modified event or None to drop the event
    """
    # Don't send health check errors
    if event.get('transaction') == 'GET /api/health':
        return None

    # Don't send expected user errors (4xx)
    if 'exception' in event:
        exc_type = event['exception']['values'][0].get('type', '')

        # Filter out expected exceptions
        expected_exceptions = [
            'NoFaceDetectedException',
            'MultipleFacesException',
            'InvalidFileFormatException'
        ]

        if exc_type in expected_exceptions:
            return None

    # Filter out rate limit test requests
    request = event.get('request', {})
    if request.get('headers', {}).get('X-Test-Request') == 'true':
        return None

    return event


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None
):
    """
    Add breadcrumb for Sentry debugging

    Args:
        message: Breadcrumb message
        category: Category (e.g., "api", "db", "cache")
        level: Log level ("debug", "info", "warning", "error")
        data: Additional context data

    Example:
        add_breadcrumb(
            "Gemini API called",
            category="api",
            level="info",
            data={"face_shape": "계란형", "latency_ms": 1250}
        )
    """
    try:
        import sentry_sdk

        sentry_sdk.add_breadcrumb(
            message=message,
            category=category,
            level=level,
            data=data or {}
        )
    except:
        # Don't let breadcrumb errors break the app
        pass


def set_context(context_name: str, context_data: Dict[str, Any]):
    """
    Set context for current Sentry scope

    Args:
        context_name: Context name (e.g., "user", "analysis", "model")
        context_data: Context data dict

    Example:
        set_context("analysis", {
            "face_shape": "계란형",
            "skin_tone": "가을웜",
            "method": "hybrid"
        })
    """
    try:
        import sentry_sdk

        sentry_sdk.set_context(context_name, context_data)
    except:
        pass


def set_user(user_id: str, **kwargs):
    """
    Set user information for Sentry

    Args:
        user_id: User identifier
        **kwargs: Additional user attributes (email, username, ip_address, etc.)

    Example:
        set_user("user_123", ip_address="127.0.0.1")
    """
    try:
        import sentry_sdk

        sentry_sdk.set_user({
            "id": user_id,
            **kwargs
        })
    except:
        pass


def capture_exception(exception: Exception, **scope_kwargs):
    """
    Manually capture an exception to Sentry

    Args:
        exception: Exception to capture
        **scope_kwargs: Additional scope attributes (tags, level, etc.)

    Example:
        try:
            risky_operation()
        except Exception as e:
            capture_exception(e, tags={"component": "ml_model"})
    """
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            # Add tags
            for key, value in scope_kwargs.get('tags', {}).items():
                scope.set_tag(key, value)

            # Set level
            if 'level' in scope_kwargs:
                scope.set_level(scope_kwargs['level'])

            sentry_sdk.capture_exception(exception)
    except:
        # Don't let Sentry errors break the app
        logger.error(f"Failed to capture exception in Sentry: {str(exception)}")


def capture_message(message: str, level: str = "info", **scope_kwargs):
    """
    Capture a message to Sentry

    Args:
        message: Message to send
        level: Log level ("debug", "info", "warning", "error", "fatal")
        **scope_kwargs: Additional scope attributes

    Example:
        capture_message(
            "High error rate detected",
            level="warning",
            tags={"component": "gemini_api"}
        )
    """
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            for key, value in scope_kwargs.get('tags', {}).items():
                scope.set_tag(key, value)

            sentry_sdk.capture_message(message, level=level)
    except:
        pass


def start_transaction(op: str, name: str, **kwargs):
    """
    Start a performance transaction

    Args:
        op: Operation type (e.g., "http.server", "db.query", "ai.inference")
        name: Transaction name
        **kwargs: Additional transaction attributes

    Returns:
        Transaction context manager

    Example:
        with start_transaction("ai.inference", "gemini_analysis"):
            result = analyze_with_gemini(image)
    """
    try:
        import sentry_sdk

        return sentry_sdk.start_transaction(op=op, name=name, **kwargs)
    except:
        # Return dummy context manager if Sentry not available
        from contextlib import nullcontext
        return nullcontext()


def start_span(op: str, description: str, **kwargs):
    """
    Start a performance span within a transaction

    Args:
        op: Operation type
        description: Span description
        **kwargs: Additional span attributes

    Returns:
        Span context manager

    Example:
        with start_transaction("api", "analyze_face"):
            with start_span("ai.inference", "gemini_api_call"):
                result = model.generate_content(prompt)
    """
    try:
        import sentry_sdk

        return sentry_sdk.start_span(op=op, description=description, **kwargs)
    except:
        from contextlib import nullcontext
        return nullcontext()


# Performance monitoring helpers

def track_performance(func_name: str, duration_ms: float, **tags):
    """
    Track performance metric

    Args:
        func_name: Function name
        duration_ms: Duration in milliseconds
        **tags: Additional tags

    Example:
        start = time.time()
        result = expensive_operation()
        track_performance("expensive_operation", (time.time() - start) * 1000)
    """
    add_breadcrumb(
        f"{func_name} completed in {duration_ms:.2f}ms",
        category="performance",
        level="info",
        data={"duration_ms": duration_ms, **tags}
    )


def track_gemini_api_call(latency_ms: float, success: bool, **context):
    """
    Track Gemini API call metrics

    Args:
        latency_ms: API call latency in milliseconds
        success: Whether the call succeeded
        **context: Additional context (face_shape, skin_tone, etc.)
    """
    add_breadcrumb(
        f"Gemini API call: {'success' if success else 'failed'} ({latency_ms:.0f}ms)",
        category="api",
        level="info" if success else "warning",
        data={
            "latency_ms": latency_ms,
            "success": success,
            **context
        }
    )
