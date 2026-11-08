"""Tests for the database package (DynamoDB-only)"""

import importlib


class TestNoLegacyDatabaseModules:
    def test_no_mysql_modules_remain(self):
        """MySQL/SQLAlchemy 시절 모듈이 다시 들어오면 실패한다"""
        for name in (
            "database.connection",
            "database.models",
            "database.mysql_repository",
            "database.migration",
            "database.repository",
        ):
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} should have been removed")
