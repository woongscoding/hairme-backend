#!/usr/bin/env python3
"""
DynamoDB Integration Test for HairMe Backend

This script tests the DynamoDB connection functions with real analysis data
to ensure compatibility with the existing MySQL-based interface.

Usage:
    # Set environment variables first
    export USE_DYNAMODB=true
    export AWS_REGION=ap-northeast-2
    export DYNAMODB_TABLE_NAME=hairme-analysis

    # Run the test
    python scripts/test_dynamodb_integration.py

Prerequisites:
    - DynamoDB table created: ./scripts/create_dynamodb_table.sh
    - AWS credentials configured: aws configure
    - boto3 installed: pip install boto3

Tests:
    1. Connection initialization
    2. Save analysis record (with MediaPipe + OpenCV data)
    3. Retrieve analysis by ID
    4. Save feedback for each style
    5. Retrieve recent analyses
    6. Get feedback statistics
"""

import sys
import os

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


def test_connection():
    """Test DynamoDB connection"""
    print("\n" + "=" * 60)
    print("Test 1: DynamoDB Connection")
    print("=" * 60)

    success = init_dynamodb()
    if success:
        print("✅ PASS: Connection initialized")
        return True
    else:
        print("❌ FAIL: Connection failed")
        return False


def test_save_analysis():
    """Test saving analysis record with full data"""
    print("\n" + "=" * 60)
    print("Test 2: Save Analysis Record")
    print("=" * 60)

    # Simulate real analysis data from analyze.py
    test_data = {
        'user_id': 'test_user_integration',
        'image_hash': 'test_hash_abc123def456',
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

    analysis_id = save_analysis(test_data)

    if analysis_id:
        print(f"✅ PASS: Analysis saved with ID: {analysis_id}")
        return analysis_id
    else:
        print("❌ FAIL: Failed to save analysis")
        return None


def test_get_analysis(analysis_id: str):
    """Test retrieving analysis by ID"""
    print("\n" + "=" * 60)
    print("Test 3: Retrieve Analysis by ID")
    print("=" * 60)

    result = get_analysis(analysis_id)

    if result:
        print(f"✅ PASS: Analysis retrieved")
        print(f"   Analysis ID: {result.get('analysis_id')}")
        print(f"   Face Shape: {result.get('face_shape')}")
        print(f"   Personal Color: {result.get('personal_color')}")
        print(f"   Detection Method: {result.get('detection_method')}")
        print(f"   MediaPipe Complete: {result.get('mediapipe_features_complete')}")
        print(f"   Processing Time: {result.get('processing_time')}s")
        return True
    else:
        print("❌ FAIL: Failed to retrieve analysis")
        return False


def test_save_feedback(analysis_id: str):
    """Test saving feedback for each style"""
    print("\n" + "=" * 60)
    print("Test 4: Save Feedback")
    print("=" * 60)

    tests = [
        (1, 'good', True),
        (2, 'bad', False),
        (3, 'good', True)
    ]

    all_passed = True

    for style_index, feedback, naver_clicked in tests:
        success = save_feedback(analysis_id, style_index, feedback, naver_clicked)

        if success:
            print(f"✅ PASS: Style {style_index} feedback saved ({feedback}, clicked={naver_clicked})")
        else:
            print(f"❌ FAIL: Style {style_index} feedback failed")
            all_passed = False

    return all_passed


def test_get_recent_analyses():
    """Test retrieving recent analyses"""
    print("\n" + "=" * 60)
    print("Test 5: Retrieve Recent Analyses")
    print("=" * 60)

    recent = get_recent_analyses(limit=5)

    if recent and len(recent) > 0:
        print(f"✅ PASS: Retrieved {len(recent)} recent analyses")
        for i, analysis in enumerate(recent, 1):
            print(f"   [{i}] {analysis.get('analysis_id')[:8]}... - {analysis.get('face_shape')} - {analysis.get('created_at')}")
        return True
    else:
        print("❌ FAIL: No recent analyses found")
        return False


def test_get_feedback_stats():
    """Test retrieving feedback statistics"""
    print("\n" + "=" * 60)
    print("Test 6: Get Feedback Statistics")
    print("=" * 60)

    stats = get_feedback_stats()

    if stats.get('success'):
        print(f"✅ PASS: Statistics retrieved")
        print(f"   Total Analysis: {stats.get('total_analysis')}")
        print(f"   Total Feedback: {stats.get('total_feedback')}")
        print(f"   Like Counts: {stats.get('like_counts')}")
        print(f"   Dislike Counts: {stats.get('dislike_counts')}")
        print(f"   Recent Feedbacks: {len(stats.get('recent_feedbacks', []))} items")
        return True
    else:
        print("❌ FAIL: Failed to retrieve statistics")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("DynamoDB Integration Test Suite")
    print("=" * 60)
    print(f"Region: {os.getenv('AWS_REGION')}")
    print(f"Table: {os.getenv('DYNAMODB_TABLE_NAME')}")
    print(f"USE_DYNAMODB: {os.getenv('USE_DYNAMODB')}")
    print("=" * 60)

    results = []

    # Test 1: Connection
    results.append(("Connection", test_connection()))

    if not results[0][1]:
        print("\n❌ Connection failed, aborting remaining tests")
        sys.exit(1)

    # Test 2: Save analysis
    analysis_id = test_save_analysis()
    results.append(("Save Analysis", analysis_id is not None))

    if not analysis_id:
        print("\n❌ Save analysis failed, aborting remaining tests")
        sys.exit(1)

    # Test 3: Retrieve analysis
    results.append(("Retrieve Analysis", test_get_analysis(analysis_id)))

    # Test 4: Save feedback
    results.append(("Save Feedback", test_save_feedback(analysis_id)))

    # Test 5: Recent analyses
    results.append(("Recent Analyses", test_get_recent_analyses()))

    # Test 6: Feedback stats
    results.append(("Feedback Stats", test_get_feedback_stats()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n🎉 All tests passed!")
        print("\nNext steps:")
        print("1. Update analyze.py to use DynamoDB:")
        print("   from database.dynamodb_connection import save_analysis")
        print("2. Update feedback.py to use DynamoDB:")
        print("   from database.dynamodb_connection import save_feedback, get_feedback_stats")
        print("3. Set USE_DYNAMODB=true in production environment")
        sys.exit(0)
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
