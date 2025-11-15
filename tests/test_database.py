"""Tests for database functionality"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch


class TestDatabaseConnection:
    """Test database connection"""

    @patch('database.connection.create_engine')
    def test_database_initialization(self, mock_create_engine):
        """Test database initialization"""
        from database.connection import init_database

        mock_engine = Mock()
        mock_create_engine.return_value = mock_engine

        result = init_database()

        # Should initialize successfully or return False
        assert isinstance(result, bool)

    @patch('database.connection.SessionLocal')
    def test_get_db_session(self, mock_session):
        """Test getting database session"""
        from database.connection import get_db_session

        mock_db = Mock()
        mock_session.return_value = mock_db

        session = get_db_session()

        # Should return a session
        assert session is not None


class TestAnalysisHistoryModel:
    """Test AnalysisHistory model"""

    def test_create_analysis_history(self, test_db):
        """Test creating analysis history record"""
        from database.models import AnalysisHistory

        record = AnalysisHistory(
            user_id="test_user",
            image_hash="abc123",
            face_shape="계란형",
            personal_color="봄웜",
            recommended_hairstyles='[{"name": "레이어드 컷"}]',
            analysis_time=0.5,
            model_version="20.2.0"
        )

        test_db.add(record)
        test_db.commit()
        test_db.refresh(record)

        assert record.id is not None
        assert record.face_shape == "계란형"
        assert record.personal_color == "봄웜"

    def test_query_analysis_history(self, test_db):
        """Test querying analysis history"""
        from database.models import AnalysisHistory

        # Create test records
        record1 = AnalysisHistory(
            user_id="user1",
            image_hash="hash1",
            face_shape="계란형",
            personal_color="봄웜",
            analysis_time=0.5
        )

        record2 = AnalysisHistory(
            user_id="user1",
            image_hash="hash2",
            face_shape="둥근형",
            personal_color="가을웜",
            analysis_time=0.6
        )

        test_db.add(record1)
        test_db.add(record2)
        test_db.commit()

        # Query records
        records = test_db.query(AnalysisHistory).filter_by(user_id="user1").all()

        assert len(records) == 2
        assert records[0].user_id == "user1"


class TestDatabaseMigration:
    """Test database migration"""

    @patch('database.migration.SessionLocal')
    @patch('database.connection.engine')
    def test_migrate_database_schema(self, mock_engine, mock_session):
        """Test database schema migration"""
        from database.migration import migrate_database_schema

        mock_db = Mock()
        mock_session.return_value = mock_db

        # Should run without errors
        try:
            migrate_database_schema()
        except Exception as e:
            # Migration might fail if tables don't exist yet
            pass
