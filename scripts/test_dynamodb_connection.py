#!/usr/bin/env python3
"""
HairMe DynamoDB Connection Test Script

Purpose:
    Test connection to the hairme-analysis DynamoDB table and perform
    basic CRUD operations to verify the table is working correctly.

Usage:
    python scripts/test_dynamodb_connection.py [--region REGION] [--table-name TABLE_NAME]

Arguments:
    --region REGION           AWS region (default: ap-northeast-2)
    --table-name TABLE_NAME   DynamoDB table name (default: hairme-analysis)

Examples:
    # Test with default settings
    python scripts/test_dynamodb_connection.py

    # Test with specific region
    python scripts/test_dynamodb_connection.py --region us-east-1

    # Test with custom table name
    python scripts/test_dynamodb_connection.py --table-name my-test-table

Prerequisites:
    - boto3 installed: pip install boto3
    - AWS credentials configured: aws configure
    - DynamoDB table created: ./scripts/create_dynamodb_table.sh

Table Schema (analysis_history → DynamoDB):
    - analysis_id (String, PK): UUID v4
    - entity_type (String): Always "ANALYSIS" (for GSI)
    - created_at (String): ISO 8601 timestamp
    - user_id (String): User identifier
    - image_hash (String): SHA-256 hash of uploaded image
    - face_shape (String): Detected face shape
    - personal_color (String): Personal color analysis result
    - recommendations (Map): JSON object with recommended styles
    - processing_time (Number): Analysis processing time in seconds
    - detection_method (String): Detection method used (opencv/mediapipe/gemini)

    OpenCV Measurements (10 attributes):
    - opencv_face_ratio, opencv_forehead_ratio, opencv_cheekbone_ratio, opencv_jaw_ratio
    - opencv_prediction, opencv_confidence, opencv_gemini_agreement
    - opencv_upper_face_ratio, opencv_middle_face_ratio, opencv_lower_face_ratio

    MediaPipe Measurements (10 attributes):
    - mediapipe_face_ratio, mediapipe_forehead_width, mediapipe_cheekbone_width
    - mediapipe_jaw_width, mediapipe_forehead_ratio, mediapipe_jaw_ratio
    - mediapipe_ITA_value, mediapipe_hue_value, mediapipe_confidence
    - mediapipe_features_complete

    Feedback Data (7 attributes):
    - recommended_styles (Map): Detailed style recommendations
    - style_1_feedback, style_2_feedback, style_3_feedback (String): 'good' or 'bad'
    - style_1_naver_clicked, style_2_naver_clicked, style_3_naver_clicked (Boolean)
    - feedback_at (String): ISO 8601 timestamp
"""

import argparse
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    print("❌ Error: boto3 not installed")
    print("Install: pip install boto3")
    sys.exit(1)


class DynamoDBTester:
    """DynamoDB connection and operations tester"""

    def __init__(self, table_name: str, region: str):
        self.table_name = table_name
        self.region = region
        self.dynamodb = None
        self.table = None

    def connect(self) -> bool:
        """Initialize DynamoDB connection"""
        print(f"🔌 Connecting to DynamoDB in {self.region}...")
        try:
            self.dynamodb = boto3.resource('dynamodb', region_name=self.region)
            self.table = self.dynamodb.Table(self.table_name)
            print(f"✅ Connected to table: {self.table_name}")
            return True
        except NoCredentialsError:
            print("❌ AWS credentials not configured")
            print("Run: aws configure")
            return False
        except Exception as e:
            print(f"❌ Connection failed: {str(e)}")
            return False

    def describe_table(self) -> bool:
        """Get and display table metadata"""
        print("\n📋 Table Information:")
        print("=" * 60)
        try:
            response = self.table.meta.client.describe_table(TableName=self.table_name)
            table_info = response['Table']

            print(f"Table Name:        {table_info['TableName']}")
            print(f"Status:            {table_info['TableStatus']}")
            print(f"Item Count:        {table_info['ItemCount']:,}")
            print(f"Size (bytes):      {table_info['TableSizeBytes']:,}")
            print(f"Created:           {table_info['CreationDateTime']}")
            print(f"Billing Mode:      {table_info.get('BillingModeSummary', {}).get('BillingMode', 'PROVISIONED')}")

            # Primary Key
            print("\n🔑 Primary Key:")
            for key in table_info['KeySchema']:
                attr_type = next(
                    attr['AttributeType']
                    for attr in table_info['AttributeDefinitions']
                    if attr['AttributeName'] == key['AttributeName']
                )
                print(f"  - {key['AttributeName']} ({attr_type}): {key['KeyType']}")

            # Global Secondary Indexes
            if 'GlobalSecondaryIndexes' in table_info:
                print("\n📊 Global Secondary Indexes:")
                for gsi in table_info['GlobalSecondaryIndexes']:
                    print(f"  - {gsi['IndexName']} ({gsi['IndexStatus']})")
                    for key in gsi['KeySchema']:
                        print(f"    - {key['AttributeName']}: {key['KeyType']}")

            # Point-in-Time Recovery
            pitr_response = self.table.meta.client.describe_continuous_backups(
                TableName=self.table_name
            )
            pitr_status = pitr_response['ContinuousBackupsDescription']['PointInTimeRecoveryDescription']['PointInTimeRecoveryStatus']
            print(f"\n💾 Point-in-Time Recovery: {pitr_status}")

            print("=" * 60)
            return True

        except ClientError as e:
            print(f"❌ Error describing table: {e.response['Error']['Message']}")
            return False

    def create_test_item(self) -> Optional[str]:
        """Create a test analysis record"""
        print("\n✏️  Creating test item...")

        analysis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        item = {
            # Primary Key & GSI
            'analysis_id': analysis_id,
            'entity_type': 'ANALYSIS',  # For GSI partition key
            'created_at': now,

            # Basic Information
            'user_id': 'test_user_' + str(uuid.uuid4())[:8],
            'image_hash': 'test_hash_' + str(uuid.uuid4()),
            'face_shape': '계란형',
            'personal_color': '봄웜',
            'recommendations': {
                'hairstyles': ['레이어드 컷', '시스루 뱅', '웨이브 펌'],
                'colors': ['애쉬 브라운', '베이지 블론드']
            },
            'processing_time': 2.5,
            'detection_method': 'mediapipe',

            # OpenCV Measurements
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

            # MediaPipe Measurements
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

            # Feedback (initially null/false)
            'recommended_styles': {
                'style_1': {'name': '레이어드 컷', 'url': 'https://example.com/1'},
                'style_2': {'name': '시스루 뱅', 'url': 'https://example.com/2'},
                'style_3': {'name': '웨이브 펌', 'url': 'https://example.com/3'}
            },
            'style_1_feedback': None,
            'style_2_feedback': None,
            'style_3_feedback': None,
            'style_1_naver_clicked': False,
            'style_2_naver_clicked': False,
            'style_3_naver_clicked': False,
            'feedback_at': None
        }

        try:
            self.table.put_item(Item=item)
            print(f"✅ Test item created: {analysis_id}")
            print(f"   User ID: {item['user_id']}")
            print(f"   Face Shape: {item['face_shape']}")
            print(f"   Created At: {now}")
            return analysis_id
        except ClientError as e:
            print(f"❌ Failed to create item: {e.response['Error']['Message']}")
            return None

    def read_test_item(self, analysis_id: str) -> bool:
        """Read the test item by analysis_id"""
        print(f"\n📖 Reading item: {analysis_id}...")

        try:
            response = self.table.get_item(Key={'analysis_id': analysis_id})

            if 'Item' not in response:
                print("❌ Item not found")
                return False

            item = response['Item']
            print("✅ Item retrieved successfully:")
            print(f"   Analysis ID: {item['analysis_id']}")
            print(f"   User ID: {item['user_id']}")
            print(f"   Face Shape: {item['face_shape']}")
            print(f"   Personal Color: {item['personal_color']}")
            print(f"   Processing Time: {item['processing_time']}s")
            print(f"   Detection Method: {item['detection_method']}")
            print(f"   MediaPipe Complete: {item['mediapipe_features_complete']}")
            return True

        except ClientError as e:
            print(f"❌ Failed to read item: {e.response['Error']['Message']}")
            return False

    def update_test_item(self, analysis_id: str) -> bool:
        """Update the test item with feedback"""
        print(f"\n✏️  Updating item with feedback: {analysis_id}...")

        try:
            response = self.table.update_item(
                Key={'analysis_id': analysis_id},
                UpdateExpression="""
                    SET style_1_feedback = :feedback,
                        style_1_naver_clicked = :clicked,
                        feedback_at = :timestamp
                """,
                ExpressionAttributeValues={
                    ':feedback': 'good',
                    ':clicked': True,
                    ':timestamp': datetime.now(timezone.utc).isoformat()
                },
                ReturnValues='ALL_NEW'
            )

            item = response['Attributes']
            print("✅ Item updated successfully:")
            print(f"   Style 1 Feedback: {item['style_1_feedback']}")
            print(f"   Style 1 Clicked: {item['style_1_naver_clicked']}")
            print(f"   Feedback At: {item['feedback_at']}")
            return True

        except ClientError as e:
            print(f"❌ Failed to update item: {e.response['Error']['Message']}")
            return False

    def query_by_time(self) -> bool:
        """Query items using created_at-index GSI"""
        print("\n🔍 Querying recent items using GSI (created_at-index)...")

        try:
            response = self.table.query(
                IndexName='created_at-index',
                KeyConditionExpression='entity_type = :type',
                ExpressionAttributeValues={
                    ':type': 'ANALYSIS'
                },
                ScanIndexForward=False,  # Descending order (newest first)
                Limit=5
            )

            items = response.get('Items', [])
            print(f"✅ Found {len(items)} items (showing up to 5 most recent):")

            for i, item in enumerate(items, 1):
                print(f"\n   [{i}] Analysis ID: {item['analysis_id']}")
                print(f"       Created: {item['created_at']}")
                print(f"       User: {item.get('user_id', 'N/A')}")
                print(f"       Face Shape: {item.get('face_shape', 'N/A')}")

            return True

        except ClientError as e:
            print(f"❌ Query failed: {e.response['Error']['Message']}")
            return False

    def delete_test_item(self, analysis_id: str) -> bool:
        """Delete the test item"""
        print(f"\n🗑️  Deleting test item: {analysis_id}...")

        try:
            self.table.delete_item(Key={'analysis_id': analysis_id})
            print("✅ Item deleted successfully")
            return True
        except ClientError as e:
            print(f"❌ Failed to delete item: {e.response['Error']['Message']}")
            return False

    def run_all_tests(self) -> bool:
        """Run all CRUD tests"""
        print("\n" + "=" * 60)
        print("🧪 Running DynamoDB CRUD Tests")
        print("=" * 60)

        # Connect
        if not self.connect():
            return False

        # Describe table
        if not self.describe_table():
            return False

        # Create
        analysis_id = self.create_test_item()
        if not analysis_id:
            return False

        # Read
        if not self.read_test_item(analysis_id):
            return False

        # Update
        if not self.update_test_item(analysis_id):
            return False

        # Query by time
        if not self.query_by_time():
            return False

        # Delete
        if not self.delete_test_item(analysis_id):
            return False

        # Verify deletion
        print(f"\n✔️  Verifying deletion...")
        if self.read_test_item(analysis_id):
            print("⚠️  Warning: Item still exists after deletion")
            return False

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Test DynamoDB connection for HairMe analysis table',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--region',
        default='ap-northeast-2',
        help='AWS region (default: ap-northeast-2)'
    )
    parser.add_argument(
        '--table-name',
        default='hairme-analysis',
        help='DynamoDB table name (default: hairme-analysis)'
    )

    args = parser.parse_args()

    tester = DynamoDBTester(table_name=args.table_name, region=args.region)

    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
