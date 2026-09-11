"""
Pytest test suite for DynamoDB integration

이 스위트는 실제 DynamoDB 테이블에 **쓰기**를 수행한다. 따라서 기본적으로
항상 skip 되며, 아래 두 가지를 모두 명시적으로 지정했을 때만 실행된다.

    RUN_DYNAMODB_INTEGRATION=1
    DYNAMODB_TEST_TABLE_NAME=hairme-analysis-test

과거에는 "자격증명이 있으면" 실행되었고 테이블 이름이 기본값
hairme-analysis(운영) 로 해석되어, 로컬이나 CI 에서 전체 스위트를 돌리는
것만으로 운영 테이블에 테스트 레코드가 쌓였다. 지금은 운영 테이블 이름이
지정되면 실행 자체를 거부한다.

Usage:
    # 테스트 전용 테이블 생성 (1회)
    ./scripts/create_test_table.sh

    # 실행
    RUN_DYNAMODB_INTEGRATION=1 \
    DYNAMODB_TEST_TABLE_NAME=hairme-analysis-test \
    pytest tests/test_dynamodb_integration.py -v

Prerequisites:
    - 테스트 테이블 생성: ./scripts/create_test_table.sh
    - AWS credentials configured: aws configure
    - boto3 installed: pip install boto3
"""

import os
import sys
import time
import uuid
import pytest
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 운영 테이블 - 이 스위트가 절대 건드리면 안 되는 이름
PRODUCTION_TABLE_NAME = "hairme-analysis"

# 명시적 옵트인 플래그
RUN_INTEGRATION_ENV = "RUN_DYNAMODB_INTEGRATION"
TEST_TABLE_ENV = "DYNAMODB_TEST_TABLE_NAME"


def _test_table_name() -> str:
    return os.getenv(TEST_TABLE_ENV, "").strip()


def _integration_skip_reason() -> str:
    """
    실행 조건을 만족하지 못한 이유를 반환한다 (조건 충족 시 빈 문자열).

    자격증명 유무는 실행 조건이 아니다. 반드시 두 환경변수를 명시해야 하고,
    지정된 테이블이 운영 테이블이면 그 자리에서 거부한다.
    """
    if os.getenv(RUN_INTEGRATION_ENV) != "1":
        return (
            f"{RUN_INTEGRATION_ENV}=1 이 아니므로 skip "
            "(실제 DynamoDB 테이블에 쓰는 통합 테스트)"
        )

    table_name = _test_table_name()
    if not table_name:
        return f"{TEST_TABLE_ENV} 이 설정되지 않아 skip (전용 테스트 테이블 필요)"

    if table_name == PRODUCTION_TABLE_NAME:
        return (
            f"{TEST_TABLE_ENV}={table_name} 은 운영 테이블이므로 거부 "
            "(예: hairme-analysis-test 를 사용)"
        )

    try:
        import boto3

        region = os.getenv("AWS_REGION", "ap-northeast-2")
        dynamodb = boto3.resource("dynamodb", region_name=region)
        dynamodb.Table(table_name).load()
    except Exception as e:
        return f"테스트 테이블 {table_name} 에 접근할 수 없어 skip: {e}"

    return ""


_SKIP_REASON = _integration_skip_reason()

# 조건을 전부 만족할 때만 실행한다
pytestmark = pytest.mark.skipif(bool(_SKIP_REASON), reason=_SKIP_REASON)

import database.dynamodb_connection as ddb
from database.dynamodb_connection import (
    init_dynamodb,
    save_analysis,
    get_analysis,
    save_feedback,
    get_recent_analyses,
    get_feedback_stats,
)

# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def dynamodb_connection():
    """
    DynamoDB 연결을 이 모듈에서만 활성화한다.

    과거에는 모듈 import 시점에 os.environ["USE_DYNAMODB"]="true" 를 설정했는데,
    수집(collection) 단계에서 실행되므로 세션 전체 환경을 오염시켜
    다른 테스트 모듈(TestClient startup -> init_database)까지 DynamoDB 분기를
    타게 만들었다. 여기서는 fixture 안에서만 환경을 바꾸고, 모듈 전역
    (dynamodb_resource/table/enabled)도 원래 값으로 되돌린다.

    테이블 이름은 DYNAMODB_TEST_TABLE_NAME 에서만 온다. 기본값으로
    운영 테이블로 흘러갈 여지를 남기지 않는다.
    """
    saved_globals = (
        ddb.dynamodb_resource,
        ddb.dynamodb_table,
        ddb.dynamodb_enabled,
    )

    with pytest.MonkeyPatch.context() as mp:
        table_name = _test_table_name()
        assert (
            table_name and table_name != PRODUCTION_TABLE_NAME
        ), f"통합 테스트는 전용 테스트 테이블에서만 실행된다 (got {table_name!r})"

        mp.setenv("USE_DYNAMODB", "true")
        mp.setenv("AWS_REGION", os.getenv("AWS_REGION", "ap-northeast-2"))
        mp.setenv("DYNAMODB_TABLE_NAME", table_name)

        success = init_dynamodb()
        assert success, "Failed to initialize DynamoDB connection"
        try:
            yield
        finally:
            (
                ddb.dynamodb_resource,
                ddb.dynamodb_table,
                ddb.dynamodb_enabled,
            ) = saved_globals


def _save_analysis_checked(data: Dict[str, Any]) -> str:
    """save_analysis 실패(None 반환)를 그 자리에서 명확히 드러낸다"""
    analysis_id = save_analysis(data)
    assert analysis_id is not None, "save_analysis returned None (write failed)"
    return analysis_id


def _get_analysis_eventually(
    analysis_id: str, attempts: int = 5, delay: float = 0.3
) -> Dict[str, Any]:
    """
    방금 쓴 항목을 다시 읽는다 (read-after-write).

    get_analysis() 는 ConsistentRead 를 쓰지 않으므로, 쓰기 직후 조회가
    간헐적으로 None 을 반환할 수 있다. 예전 코드는 그 None 을 그대로
    result["..."] 로 첨자 접근해서 전체 스위트 실행 시 간헐적인
    TypeError: 'NoneType' object is not subscriptable 로 터졌다.
    여기서는 짧게 재조회한 뒤에도 없으면 원인이 드러나는 assert 로 실패시킨다.
    """
    result = None
    for attempt in range(attempts):
        result = get_analysis(analysis_id)
        if result is not None:
            return result
        time.sleep(delay * (attempt + 1))

    raise AssertionError(
        f"analysis {analysis_id} not readable after {attempts} attempts "
        "(eventually consistent read never converged)"
    )


@pytest.fixture(scope="module")
def shared_analysis_id(dynamodb_connection) -> str:
    """
    여러 테스트가 공유하는 분석 레코드.

    과거에는 pytest 모듈 객체에 pytest.shared_analysis_id 를 붙여 테스트 간
    순서 의존성을 만들었다. fixture 로 바꿔 어떤 테스트를 단독 실행해도
    동작하게 한다.
    """
    data = {
        "user_id": f"test_user_{uuid.uuid4().hex[:8]}",
        "image_hash": f"test_hash_{uuid.uuid4().hex}",
        "face_shape": "계란형",
        "personal_color": "봄웜",
        "recommendations": [
            {"style_name": "레이어드 컷", "reason": "얼굴형에 잘 어울림"},
        ],
        "processing_time": 2.5,
        "detection_method": "mediapipe",
        "opencv_confidence": 0.87,
        "mediapipe_face_ratio": 1.28,
        "mediapipe_ITA_value": 28.5,
        "mediapipe_features_complete": True,
    }
    analysis_id = _save_analysis_checked(data)
    _get_analysis_eventually(analysis_id)
    return analysis_id


@pytest.fixture
def sample_analysis_data() -> Dict[str, Any]:
    """Sample analysis data matching real API structure"""
    return {
        "user_id": f"test_user_{uuid.uuid4().hex[:8]}",
        "image_hash": f"test_hash_{uuid.uuid4().hex}",
        "face_shape": "계란형",
        "personal_color": "봄웜",
        "recommendations": [
            {"style_name": "레이어드 컷", "reason": "얼굴형에 잘 어울림"},
            {"style_name": "시스루 뱅", "reason": "이마가 넓어서 추천"},
            {"style_name": "웨이브 펌", "reason": "부드러운 인상"},
        ],
        "processing_time": 2.5,
        "detection_method": "mediapipe",
        # OpenCV measurements
        "opencv_face_ratio": 1.3,
        "opencv_forehead_ratio": 0.85,
        "opencv_cheekbone_ratio": 0.92,
        "opencv_jaw_ratio": 0.78,
        "opencv_prediction": "계란형",
        "opencv_confidence": 0.87,
        "opencv_gemini_agreement": True,
        "opencv_upper_face_ratio": 0.33,
        "opencv_middle_face_ratio": 0.34,
        "opencv_lower_face_ratio": 0.33,
        # MediaPipe measurements
        "mediapipe_face_ratio": 1.28,
        "mediapipe_forehead_width": 145.5,
        "mediapipe_cheekbone_width": 158.2,
        "mediapipe_jaw_width": 122.8,
        "mediapipe_forehead_ratio": 0.92,
        "mediapipe_jaw_ratio": 0.78,
        "mediapipe_ITA_value": 28.5,
        "mediapipe_hue_value": 15.2,
        "mediapipe_confidence": 0.94,
        "mediapipe_features_complete": True,
        # Recommended styles (detailed)
        "recommended_styles": [
            {
                "style_name": "레이어드 컷",
                "reason": "얼굴형에 잘 어울림",
                "url": "https://example.com/1",
            },
            {
                "style_name": "시스루 뱅",
                "reason": "이마가 넓어서 추천",
                "url": "https://example.com/2",
            },
            {
                "style_name": "웨이브 펌",
                "reason": "부드러운 인상",
                "url": "https://example.com/3",
            },
        ],
    }


# ==================== Connection Tests ====================


def test_init_dynamodb(dynamodb_connection):
    """Test DynamoDB connection initialization"""
    # Connection already initialized by fixture
    assert True, "DynamoDB connection initialized successfully"


# ==================== CRUD Tests ====================


def test_save_analysis(dynamodb_connection, sample_analysis_data):
    """Test saving analysis record with full data"""
    analysis_id = _save_analysis_checked(sample_analysis_data)

    assert isinstance(analysis_id, str), "analysis_id should be a string (UUID)"
    assert len(analysis_id) == 36, "analysis_id should be UUID v4 format"


def test_get_analysis(shared_analysis_id):
    """Test retrieving analysis by ID"""
    result = _get_analysis_eventually(shared_analysis_id)

    assert result["analysis_id"] == shared_analysis_id
    assert result["face_shape"] == "계란형"
    assert result["personal_color"] == "봄웜"
    assert result["detection_method"] == "mediapipe"
    assert result["mediapipe_features_complete"] is True

    # Verify MediaPipe data
    assert "mediapipe_face_ratio" in result
    assert "mediapipe_ITA_value" in result
    assert isinstance(result["mediapipe_face_ratio"], float)

    # Verify OpenCV data
    assert "opencv_confidence" in result
    assert isinstance(result["opencv_confidence"], float)


def test_get_analysis_not_found(dynamodb_connection):
    """Test retrieving non-existent analysis"""
    fake_id = str(uuid.uuid4())
    result = get_analysis(fake_id)

    assert result is None, "get_analysis should return None for non-existent ID"


def test_save_feedback_style_1(shared_analysis_id):
    """Test saving feedback for style 1"""
    analysis_id = shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id, style_index=1, feedback="good", naver_clicked=True
    )

    assert success is True, "save_feedback should return True"

    # Verify feedback was saved
    result = _get_analysis_eventually(analysis_id)
    assert result["style_1_feedback"] == "good"
    assert result["style_1_naver_clicked"] is True
    assert result["feedback_at"] is not None


def test_save_feedback_style_2(shared_analysis_id):
    """Test saving feedback for style 2"""
    analysis_id = shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id, style_index=2, feedback="bad", naver_clicked=False
    )

    assert success is True

    result = _get_analysis_eventually(analysis_id)
    assert result["style_2_feedback"] == "bad"
    assert result["style_2_naver_clicked"] is False


def test_save_feedback_invalid_style_index(shared_analysis_id):
    """Test saving feedback with invalid style index"""
    analysis_id = shared_analysis_id

    success = save_feedback(
        analysis_id=analysis_id,
        style_index=99,  # Invalid
        feedback="good",
        naver_clicked=True,
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
        assert "analysis_id" in analysis
        assert "created_at" in analysis
        assert "face_shape" in analysis

    # Verify order (newest first)
    if len(recent) >= 2:
        assert (
            recent[0]["created_at"] >= recent[1]["created_at"]
        ), "Results should be sorted by created_at descending"


def test_get_feedback_stats(shared_analysis_id):
    """Test retrieving feedback statistics"""
    # 순서 의존을 없애기 위해 통계에 잡힐 피드백을 이 테스트가 직접 만든다
    assert save_feedback(shared_analysis_id, 1, "good", True) is True

    stats = get_feedback_stats()

    assert stats["success"] is True
    assert "total_analysis" in stats
    assert "total_feedback" in stats
    assert "like_counts" in stats
    assert "dislike_counts" in stats
    assert "recent_feedbacks" in stats

    # Verify counts structure
    assert isinstance(stats["like_counts"], dict)
    assert "style_1" in stats["like_counts"]
    assert "style_2" in stats["like_counts"]
    assert "style_3" in stats["like_counts"]

    # Verify we have at least 1 feedback (from previous tests)
    assert (
        stats["total_feedback"] > 0
    ), "Should have at least 1 feedback from previous tests"
    assert (
        stats["like_counts"]["style_1"] > 0
    ), "Should have at least 1 like for style_1"


# ==================== Data Type Tests ====================


def test_float_decimal_conversion(dynamodb_connection, sample_analysis_data):
    """Test that floats are properly converted to/from Decimal"""
    # Save with float values
    analysis_id = _save_analysis_checked(sample_analysis_data)

    # Retrieve and verify types (read-after-write 는 즉시 보이지 않을 수 있다)
    result = _get_analysis_eventually(analysis_id)

    # Should be converted back to float
    assert isinstance(result["mediapipe_face_ratio"], float)
    assert isinstance(result["mediapipe_ITA_value"], float)
    assert isinstance(result["opencv_confidence"], float)
    assert isinstance(result["processing_time"], float)


def test_null_values_handling(dynamodb_connection):
    """Test handling of null/optional values"""
    minimal_data = {
        "image_hash": f"test_minimal_{uuid.uuid4().hex}",
        "face_shape": "둥근형",
        "personal_color": "여름쿨",
        "recommendations": [],
        "processing_time": 1.0,
        "detection_method": "gemini",
        # No MediaPipe or OpenCV data
    }

    analysis_id = _save_analysis_checked(minimal_data)

    result = _get_analysis_eventually(analysis_id)
    assert result["face_shape"] == "둥근형"
    assert (
        result.get("mediapipe_face_ratio") is None
        or "mediapipe_face_ratio" not in result
    )


# ==================== Error Handling Tests ====================


def test_save_feedback_nonexistent_analysis(dynamodb_connection):
    """Test saving feedback for non-existent analysis"""
    fake_id = str(uuid.uuid4())

    success = save_feedback(
        analysis_id=fake_id, style_index=1, feedback="good", naver_clicked=True
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
        data["image_hash"] = f"batch_test_{i}_{uuid.uuid4().hex}"
        analysis_ids.append(_save_analysis_checked(data))

    elapsed = time.time() - start_time

    assert len(analysis_ids) == 5
    assert elapsed < 10.0, f"Saving 5 records should take < 10s (took {elapsed:.2f}s)"

    print(f"\nSaved 5 records in {elapsed:.2f}s ({elapsed/5:.2f}s per record)")


# ==================== Integration Tests ====================


def test_full_analysis_workflow(dynamodb_connection, sample_analysis_data):
    """Test complete analysis workflow: create -> retrieve -> feedback -> stats"""
    # 1. Create analysis
    analysis_id = _save_analysis_checked(sample_analysis_data)

    # 2. Retrieve analysis
    result = _get_analysis_eventually(analysis_id)
    assert result["analysis_id"] == analysis_id

    # 3. Add feedback for all 3 styles
    for i in range(1, 4):
        feedback_value = "good" if i % 2 == 1 else "bad"
        success = save_feedback(analysis_id, i, feedback_value, True)
        assert success is True

    # 4. Verify feedback was saved
    result = _get_analysis_eventually(analysis_id)
    assert result["style_1_feedback"] == "good"
    assert result["style_2_feedback"] == "bad"
    assert result["style_3_feedback"] == "good"

    # 5. Check stats
    stats = get_feedback_stats()
    assert stats["success"] is True
    assert stats["total_feedback"] > 0

    print(f"\nFull workflow test passed: {analysis_id}")


# ==================== Cleanup ====================


def test_cleanup_test_data(shared_analysis_id):
    """
    Note: DynamoDB doesn't require explicit cleanup in tests.
    Test data will remain in the table for verification.
    To clean up manually, use AWS Console or CLI.
    """
    result = _get_analysis_eventually(shared_analysis_id)
    assert result is not None, "Test data should still exist"

    print(f"\nTest data preserved: {shared_analysis_id}")
    print("To clean up test data, use AWS Console or delete-item CLI command")


# ==================== Main ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
