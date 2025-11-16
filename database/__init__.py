"""
Database module for HairMe Backend

This module provides a unified interface for database operations,
supporting both MySQL (RDS) and DynamoDB backends.

Environment Variables:
    USE_DYNAMODB: Set to 'true' to use DynamoDB, 'false' for MySQL (default: false)

Usage:
    from database import init_database

    # Initialize database (MySQL or DynamoDB based on USE_DYNAMODB)
    init_database()

    # For MySQL:
    from database.connection import get_db_session
    db = get_db_session()

    # For DynamoDB:
    from database.dynamodb_connection import save_analysis, get_analysis
    analysis_id = save_analysis({...})
"""

import os
from core.logging import logger


def init_database() -> bool:
    """
    Initialize database connection (MySQL or DynamoDB based on USE_DYNAMODB)

    Returns:
        bool: True if initialization successful, False otherwise
    """
    use_dynamodb = os.getenv('USE_DYNAMODB', 'false').lower() == 'true'

    if use_dynamodb:
        logger.info("🔄 Initializing DynamoDB connection...")
        from database.dynamodb_connection import init_dynamodb
        return init_dynamodb()
    else:
        logger.info("🔄 Initializing MySQL connection...")
        from database.connection import init_database as init_mysql
        return init_mysql()
