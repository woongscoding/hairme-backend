"""Tests for the database package (DynamoDB-only)"""

from unittest.mock import patch


class TestDatabaseInit:
    """database.init_database() must initialise DynamoDB only"""

    @patch("database.dynamodb_connection.init_dynamodb")
    def test_init_database_delegates_to_dynamodb(self, mock_init):
        from database import init_database

        mock_init.return_value = True

        result = init_database()

        assert result is True
        mock_init.assert_called_once_with()

    @patch("database.dynamodb_connection.init_dynamodb")
    def test_init_database_returns_bool_on_failure(self, mock_init):
        from database import init_database

        mock_init.return_value = False

        assert init_database() is False

    def test_no_mysql_modules_remain(self):
        """MySQL/SQLAlchemy 경로가 다시 들어오면 실패한다"""
        import importlib

        for name in (
            "database.connection",
            "database.models",
            "database.mysql_repository",
        ):
            try:
                importlib.import_module(name)
            except ImportError:
                continue
            raise AssertionError(f"{name} should have been removed")


class TestRepositoryFactory:
    """database.repository.get_repository() must always return DynamoDB"""

    def test_get_repository_returns_dynamodb_repository(self):
        from database.repository import get_repository
        from database.dynamodb_repository import DynamoDBAnalysisRepository

        assert isinstance(get_repository(), DynamoDBAnalysisRepository)

    def test_get_repository_ignores_use_dynamodb_flag(self, monkeypatch):
        from database.repository import get_repository
        from database.dynamodb_repository import DynamoDBAnalysisRepository

        monkeypatch.setenv("USE_DYNAMODB", "false")

        assert isinstance(get_repository(), DynamoDBAnalysisRepository)


class TestMigrationShim:
    """migrate_database_schema() 는 하위 호환용 no-op 이어야 한다"""

    def test_migrate_database_schema_is_noop(self):
        from database.migration import migrate_database_schema

        with patch("database.migration.logger") as mock_logger:
            assert migrate_database_schema() is True
            assert mock_logger.warning.called
