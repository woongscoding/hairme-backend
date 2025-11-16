"""
Pytest test suite for DynamoDB integration

Tests the DynamoDB connection functions and API endpoints to ensure
compatibility with the existing MySQL-based interface.

Usage:
    # Set environment variables
    export USE_DYNAMODB=true
    export AWS_REGION=ap-northeast-2
    export DYNAMODB_TABLE_NAME=hairme-analysis

    # Run all tests
    pytest tests/test_dynamodb_integration.py -v

    # Run specific test
    pytest tests/test_dynamodb_integration.py::test_save_analysis -v

Prerequisites:
    - DynamoDB table created: ./scripts/create_dynamodb_table.sh
    - AWS credentials configured: aws configure
    - boto3 installed: pip install boto3
    - pytest installed: pip install pytest pytest-asyncio
"""

import os
import sys
import uuid
import pytest
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set environment variables for testing
os.environ['USE_DYNAMODB'] = 'true'
os.environ['AWS_REGION'] = os.getenv('AWS_REGION', 'ap-northeast-2')
os.environ['DYNAMODB_TABLE_NAME'] = os.getenv('DYNAMODB_TABLE_NAME', 'hairme-analysis')

from database.dynamodb_connection import (
    init_dynamodb,
    save_analysis,
    get_analysis,
    save_feedback,
    get_recent_analyses,
    get_feedback_stats
)


# ==================== Fixtures ====================

@pytest.fixture(scope="module")
def dynamodb_connection():
    """Initialize DynamoDB connection for all tests"""
    success = init_dynamodb()
    assert success, "Failed to initialize DynamoDB connection"
    yield
    # Cleanup is handled automatically


@pytest.fixture
def sample_analysis_data() -> Dict[str, Any]:
    """Sample analysis data matching real API structure"""
    return {
        'user_id': f'test_user_{uuid.uuid4().hex[:8]}',
        'image_hash': f'test_hash_{uuid.uuid4().hex}',
        'face_shape': '계란형',
        'personal_color': '봄웜',
        'recommendations': [
            {'style_name': '레이어드 컷', 'reason': '얼굴형에 잘 어울림'},
            {'style_name': '시스루 뱅', 'reason': '이마가 넓어서 추천'},
            {'style_name': '웨이브 펌', 'reason': '부드러운 인상'}
        ],
        'processing_time': 2.5,
        'detection_method': 'mediapipe',

        # OpenCV measurements
        'opencv_face_ratio': 1.3,
        'opencv_forehead_ratio': 0.85,
        'opencv_cheekbone_ratio': 0.92,
        'opencv_jaw_ratio': 0.78,
        'opencv_prediction': '계란형',
        'opencv_confidence': 0.87,
        'opencv_gemini_agreement': True,
        'opencv_upper_face_ratio': 0.33,
        'opencv_middle_face_ratio': 0.34,
        'opencv_lower_face_ratio': 0.33,

        # MediaPipe measurements
        'mediapipe_face_ratio': 1.28,
        'mediapipe_forehead_width': 145.5,
        'mediapipe_cheekbone_width': 158.2,
        'mediapipe_jaw_width': 122.8,
        'mediapipe_forehead_ratio': 0.92,
        'mediapipe_jaw_ratio': 0.78,
        'mediapipe_ITA_value': 28.5,
        'mediapipe_hue_value': 15.2,
        'mediapipe_confidence': 0.94,
        'mediapipe_features_complete': True,

        # Recommended styles (detailed)
        'recommended_styles': [
            {'style_name': '레이어드 컷', 'reason': '얼굴형에 잘 어울림', 'url': 'https://example.com/1'},
            {'style_name': '시스루 뱅', 'reason': '이마가 넓어서 추천', 'url': 'https://example.com/2'},
            {'style_name': '웨이브 펌', 'reason': '부드러운 인상', 'url': 'https://example.com/3'}
        ]
    }


# ==================== Connection Tests ====================

def test_init_dynamodb(dynamodb_connection):
    """Test DynamoDB connection initialization"""
    # Connection already initialized by fixture
    assert True, "DynamoDB connection initialized successfully"


# ==================== CRUD Tests ====================

def test_save_analysis(dynamodb_connection, sample_analysis_data):
    """Test saving analysis record with full data"""
    analysis_id = save_analysis(sample_analysis_data)

    assert analysis_id is not None, "save_analysis should return analysis_id"
    assert isinstance(analysis_id, str), "analysis_id should be a string (UUID)"
    assert len(analysis_id) == 36, "analysis_id should be UUID v4 format"

    # Store for later tests
    pytest.shared_analysis_id = analysis_id


def test_get_analysis(dynamodb_connection):
    """Test retrieving analysis by ID"""
    analysis_id = pytest.shared_analysis_id

    result = get_analysis(analysis_id)

    assert result is not None, "get_analysis should return data"
    assert result['analysis_id'] == analysis_id
    assert result['face_shape'] == '계란형'
    assert result['personal_color'] == '봄웜'
    assert result['detection_method'] == 'mediapipe'
    assert result['mediapipe_features_complete'] is True

    # Verify MediaPipe data
    assert 'mediapipe_face_ratio' in result
    assert 'mediapipe_ITA_value' in result
    assert isinstance(result['mediapipe_face_ratio'], float)

    # Verify OpenCV data
    assert 'opencv_confidence' in result
    assert isinstance(result['opencv_confidence'], float)


def test_get_analysis_not_found(dynamodb_connection):
    """Test retrieving non-existent analysis"""
    fake_id = str(uuid.uuid4())
    result = get_analysis(fake_id)

    assert result is None, "get_analysis should return None for non-existent ID"


def test_save_feedback_style_1(dynamodb_connection):
    """Test saving feedback for style 1"""
    analysis_id = pytest.shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id,
        style_index=1,
        feedback='good',
        naver_clicked=True
    )

    assert success is True, "save_feedback should return True"

    # Verify feedback was saved
    result = get_analysis(analysis_id)
    assert result['style_1_feedback'] == 'good'
    assert result['style_1_naver_clicked'] is True
    assert result['feedback_at'] is not None


def test_save_feedback_style_2(dynamodb_connection):
    """Test saving feedback for style 2"""
    analysis_id = pytest.shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id,
        style_index=2,
        feedback='bad',
        naver_clicked=False
    )

    assert success is True

    result = get_analysis(analysis_id)
    assert result['style_2_feedback'] == 'bad'
    assert result['style_2_naver_clicked'] is False


def test_save_feedback_invalid_style_index(dynamodb_connection):
    """Test saving feedback with invalid style index"""
    analysis_id = pytest.shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id,
        style_index=99,  # Invalid
        feedback='good',
        naver_clicked=True
    )

    assert success is False, "save_feedback should return False for invalid style_index"


# ==================== Query Tests ====================

def test_get_recent_analyses(dynamodb_connection):
    """Test retrieving recent analyses"""
    recent = get_recent_analyses(limit=5)

    assert isinstance(recent, list), "get_recent_analyses should return a list"
    assert len(recent) > 0, "Should return at least 1 analysis (from previous tests)"

    # Verify data structure
    for analysis in recent:
        assert 'analysis_id' in analysis
        assert 'created_at' in analysis
        assert 'face_shape' in analysis

    # Verify order (newest first)
    if len(recent) >= 2:
        assert recent[0]['created_at'] >= recent[1]['created_at'], \
            "Results should be sorted by created_at descending"


def test_get_feedback_stats(dynamodb_connection):
    """Test retrieving feedback statistics"""
    stats = get_feedback_stats()

    assert stats['success'] is True
    assert 'total_analysis' in stats
    assert 'total_feedback' in stats
    assert 'like_counts' in stats
    assert 'dislike_counts' in stats
    assert 'recent_feedbacks' in stats

    # Verify counts structure
    assert isinstance(stats['like_counts'], dict)
    assert 'style_1' in stats['like_counts']
    assert 'style_2' in stats['like_counts']
    assert 'style_3' in stats['like_counts']

    # Verify we have at least 1 feedback (from previous tests)
    assert stats['total_feedback'] > 0, "Should have at least 1 feedback from previous tests"
    assert stats['like_counts']['style_1'] > 0, "Should have at least 1 like for style_1"


# ==================== Data Type Tests ====================

def test_float_decimal_conversion(dynamodb_connection, sample_analysis_data):
    """Test that floats are properly converted to/from Decimal"""
    # Save with float values
    analysis_id = save_analysis(sample_analysis_data)

    # Retrieve and verify types
    result = get_analysis(analysis_id)

    # Should be converted back to float
    assert isinstance(result['mediapipe_face_ratio'], float)
    assert isinstance(result['mediapipe_ITA_value'], float)
    assert isinstance(result['opencv_confidence'], float)
    assert isinstance(result['processing_time'], float)


def test_null_values_handling(dynamodb_connection):
    """Test handling of null/optional values"""
    minimal_data = {
        'image_hash': f'test_minimal_{uuid.uuid4().hex}',
        'face_shape': '둥근형',
        'personal_color': '여름쿨',
        'recommendations': [],
        'processing_time': 1.0,
        'detection_method': 'gemini'
        # No MediaPipe or OpenCV data
    }

    analysis_id = save_analysis(minimal_data)
    assert analysis_id is not None

    result = get_analysis(analysis_id)
    assert result['face_shape'] == '둥근형'
    assert result.get('mediapipe_face_ratio') is None or 'mediapipe_face_ratio' not in result


# ==================== Error Handling Tests ====================

def test_save_feedback_nonexistent_analysis(dynamodb_connection):
    """Test saving feedback for non-existent analysis"""
    fake_id = str(uuid.uuid4())

    success = save_feedback(
        analysis_id=fake_id,
        style_index=1,
        feedback='good',
        naver_clicked=True
    )

    # Should fail gracefully
    # Implementation may return False or handle differently
    # Just verify it doesn't crash
    assert isinstance(success, bool)


# ==================== Performance Tests ====================

def test_batch_save_performance(dynamodb_connection, sample_analysis_data):
    """Test saving multiple records in sequence"""
    import time

    start_time = time.time()
    analysis_ids = []

    for i in range(5):
        data = sample_analysis_data.copy()
        data['image_hash'] = f'batch_test_{i}_{uuid.uuid4().hex}'
        analysis_id = save_analysis(data)
        assert analysis_id is not None
        analysis_ids.append(analysis_id)

    elapsed = time.time() - start_time

    assert len(analysis_ids) == 5
    assert elapsed < 10.0, f"Saving 5 records should take < 10s (took {elapsed:.2f}s)"

    print(f"\n✅ Saved 5 records in {elapsed:.2f}s ({elapsed/5:.2f}s per record)")


# ==================== Integration Tests ====================

def test_full_analysis_workflow(dynamodb_connection, sample_analysis_data):
    """Test complete analysis workflow: create → retrieve → feedback → stats"""
    # 1. Create analysis
    analysis_id = save_analysis(sample_analysis_data)
    assert analysis_id is not None

    # 2. Retrieve analysis
    result = get_analysis(analysis_id)
    assert result is not None
    assert result['analysis_id'] == analysis_id

    # 3. Add feedback for all 3 styles
    for i in range(1, 4):
        feedback_value = 'good' if i % 2 == 1 else 'bad'
        success = save_feedback(analysis_id, i, feedback_value, True)
        assert success is True

    # 4. Verify feedback was saved
    result = get_analysis(analysis_id)
    assert result['style_1_feedback'] == 'good'
    assert result['style_2_feedback'] == 'bad'
    assert result['style_3_feedback'] == 'good'

    # 5. Check stats
    stats = get_feedback_stats()
    assert stats['success'] is True
    assert stats['total_feedback'] > 0

    print(f"\n✅ Full workflow test passed: {analysis_id}")


# ==================== Cleanup ====================

def test_cleanup_test_data(dynamodb_connection):
    """
    Note: DynamoDB doesn't require explicit cleanup in tests.
    Test data will remain in the table for verification.
    To clean up manually, use AWS Console or CLI.
    """
    # Get test analysis ID
    if hasattr(pytest, 'shared_analysis_id'):
        analysis_id = pytest.shared_analysis_id
        result = get_analysis(analysis_id)
        assert result is not None, "Test data should still exist"

        print(f"\n✅ Test data preserved: {analysis_id}")
        print("To clean up test data, use AWS Console or delete-item CLI command")


# ==================== Main ====================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
