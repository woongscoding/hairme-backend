"""
Database module for HairMe Backend

DynamoDB is the only supported backend. The legacy MySQL/SQLAlchemy path was
removed; ``USE_DYNAMODB`` is kept as a deployment-compatibility flag only and no
longer selects a backend.

Usage:
    from database import init_database

    init_database()

    from database.dynamodb_connection import save_analysis, get_analysis
    analysis_id = save_analysis({...})
"""

from core.logging import logger


def init_database() -> bool:
    """
    Initialize the DynamoDB connection.

    Returns:
        bool: True if initialization successful, False otherwise
    """
    logger.info("🔄 Initializing DynamoDB connection...")
    from database.dynamodb_connection import init_dynamodb

    return init_dynamodb()
