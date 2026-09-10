"""Deprecated schema-migration shim.

The MySQL/SQLAlchemy backend was removed; DynamoDB is schema-less and needs no
migration step. This module only exists so that legacy call sites (main.py's
``_init_core_services``) keep importing successfully.
"""

from core.logging import logger


def migrate_database_schema() -> bool:
    """
    No-op shim kept for backward compatibility.

    Returns:
        bool: Always True (nothing to migrate on DynamoDB).
    """
    logger.warning(
        "⚠️ migrate_database_schema() is deprecated and does nothing "
        "(DynamoDB is schema-less). Remove the call site."
    )
    return True
