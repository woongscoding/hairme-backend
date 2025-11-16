#!/usr/bin/env python3
"""
DynamoDB to RDS Rollback Script

Rolls back data from DynamoDB to RDS MySQL in case of migration issues.
This script is for EMERGENCY USE ONLY.

⚠️  WARNING: This script will:
    1. Read all data from DynamoDB
    2. Insert missing records back into RDS MySQL
    3. Update existing records if they differ

Usage:
    # Dry-run mode (recommended first)
    python scripts/rollback_to_rds.py --dry-run

    # Actual rollback
    python scripts/rollback_to_rds.py

    # Rollback specific record
    python scripts/rollback_to_rds.py --analysis-id "550e8400-..."

Prerequisites:
    - RDS MySQL accessible (DATABASE_URL, DB_PASSWORD in .env)
    - DynamoDB accessible
    - BACKUP your RDS database before running!

IMPORTANT:
    - This does NOT delete records from DynamoDB
    - This only restores records to RDS MySQL
    - Original MySQL IDs are preserved (stored in mysql_id field)

Output:
    - rollback_log_YYYYMMDD_HHMMSS.json
"""

import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from decimal import Decimal

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    print("❌ Error: pymysql not installed. Run: pip install pymysql")
    sys.exit(1)

try:
    import boto3
except ImportError:
    print("❌ Error: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()


class RollbackStats:
    """Track rollback statistics"""

    def __init__(self):
        self.total_records = 0
        self.processed = 0
        self.inserted = 0
        self.updated = 0
        self.skipped = 0
        self.failed = 0
        self.start_time = datetime.now()
        self.errors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_records': self.total_records,
            'processed': self.processed,
            'inserted': self.inserted,
            'updated': self.updated,
            'skipped': self.skipped,
            'failed': self.failed,
            'elapsed_seconds': round(elapsed, 2)
        }


def connect_to_rds() -> Optional[pymysql.Connection]:
    """Connect to RDS MySQL database"""
    database_url = os.getenv('DATABASE_URL')
    db_password = os.getenv('DB_PASSWORD')

    if not database_url:
        print("❌ Error: DATABASE_URL not set in .env")
        return None

    try:
        import urllib.parse
        from sqlalchemy.engine.url import make_url

        if "://admin@" in database_url and db_password:
            encoded_password = urllib.parse.quote_plus(db_password)
            db_url = database_url.replace("asyncmy", "pymysql").replace(
                "://admin@",
                f"://admin:{encoded_password}@"
            )
        else:
            db_url = database_url.replace("asyncmy", "pymysql")

        url = make_url(db_url)

        connection = pymysql.connect(
            host=url.host,
            port=url.port or 3306,
            user=url.username,
            password=url.password,
            database=url.database,
            cursorclass=DictCursor,
            charset='utf8mb4'
        )

        print(f"✅ Connected to RDS MySQL: {url.host}")
        return connection

    except Exception as e:
        print(f"❌ Failed to connect to RDS: {str(e)}")
        return None


def connect_to_dynamodb() -> Tuple[Optional[Any], Optional[Any]]:
    """Connect to DynamoDB"""
    aws_region = os.getenv('AWS_REGION', 'ap-northeast-2')
    table_name = os.getenv('DYNAMODB_TABLE_NAME', 'hairme-analysis')

    try:
        dynamodb = boto3.resource('dynamodb', region_name=aws_region)
        table = dynamodb.Table(table_name)
        table.load()

        print(f"✅ Connected to DynamoDB: {table_name} ({aws_region})")
        return dynamodb, table

    except Exception as e:
        print(f"❌ Failed to connect to DynamoDB: {str(e)}")
        return None, None


def fetch_dynamodb_records(table: Any) -> List[Dict[str, Any]]:
    """Fetch all records from DynamoDB"""
    print("\n📊 Fetching records from DynamoDB...")

    try:
        response = table.scan()
        items = response.get('Items', [])

        # Handle pagination
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))

        print(f"✅ Fetched {len(items):,} records from DynamoDB")
        return items

    except Exception as e:
        print(f"❌ Failed to fetch DynamoDB records: {str(e)}")
        return []


def transform_dynamodb_to_mysql(dynamodb_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform DynamoDB item back to MySQL record format

    Key transformations:
    - analysis_id (UUID) → ignore (use mysql_id if exists)
    - created_at (ISO string) → DATETIME
    - Decimal → float
    - entity_type → ignore (DynamoDB-specific)
    """
    mysql_record = {}

    # Restore original MySQL ID if available
    if 'mysql_id' in dynamodb_item:
        mysql_record['id'] = int(dynamodb_item['mysql_id'])

    # Convert created_at
    if dynamodb_item.get('created_at'):
        try:
            # Parse ISO 8601 string to datetime
            created_at_str = dynamodb_item['created_at']
            if isinstance(created_at_str, str):
                # Remove timezone if present
                created_at_str = created_at_str.replace('Z', '').split('+')[0]
                mysql_record['created_at'] = datetime.fromisoformat(created_at_str)
        except Exception as e:
            print(f"⚠️  Failed to parse created_at: {e}")
            mysql_record['created_at'] = datetime.now()

    # Basic fields
    simple_fields = [
        'user_id', 'image_hash', 'face_shape', 'personal_color',
        'processing_time', 'detection_method'
    ]
    for field in simple_fields:
        if field in dynamodb_item:
            value = dynamodb_item[field]
            # Convert Decimal to float
            if isinstance(value, Decimal):
                mysql_record[field] = float(value)
            else:
                mysql_record[field] = value

    # JSON fields
    json_fields = ['recommendations', 'recommended_styles']
    for field in json_fields:
        if field in dynamodb_item:
            value = dynamodb_item[field]
            if isinstance(value, (dict, list)):
                mysql_record[field] = json.dumps(value, ensure_ascii=False)
            else:
                mysql_record[field] = value

    # OpenCV fields
    opencv_fields = [
        'opencv_face_ratio', 'opencv_forehead_ratio', 'opencv_cheekbone_ratio',
        'opencv_jaw_ratio', 'opencv_prediction', 'opencv_confidence',
        'opencv_gemini_agreement', 'opencv_upper_face_ratio',
        'opencv_middle_face_ratio', 'opencv_lower_face_ratio'
    ]
    for field in opencv_fields:
        if field in dynamodb_item:
            value = dynamodb_item[field]
            if isinstance(value, Decimal):
                mysql_record[field] = float(value)
            else:
                mysql_record[field] = value

    # MediaPipe fields
    mediapipe_fields = [
        'mediapipe_face_ratio', 'mediapipe_forehead_width',
        'mediapipe_cheekbone_width', 'mediapipe_jaw_width',
        'mediapipe_forehead_ratio', 'mediapipe_jaw_ratio',
        'mediapipe_ITA_value', 'mediapipe_hue_value',
        'mediapipe_confidence', 'mediapipe_features_complete'
    ]
    for field in mediapipe_fields:
        if field in dynamodb_item:
            value = dynamodb_item[field]
            if isinstance(value, Decimal):
                mysql_record[field] = float(value)
            else:
                mysql_record[field] = value

    # Feedback fields
    feedback_fields = [
        'style_1_feedback', 'style_2_feedback', 'style_3_feedback',
        'style_1_naver_clicked', 'style_2_naver_clicked', 'style_3_naver_clicked'
    ]
    for field in feedback_fields:
        if field in dynamodb_item:
            mysql_record[field] = dynamodb_item[field]

    # Feedback timestamp
    if dynamodb_item.get('feedback_at'):
        try:
            feedback_at_str = dynamodb_item['feedback_at']
            if isinstance(feedback_at_str, str) and feedback_at_str != 'None':
                feedback_at_str = feedback_at_str.replace('Z', '').split('+')[0]
                mysql_record['feedback_at'] = datetime.fromisoformat(feedback_at_str)
        except Exception:
            pass

    return mysql_record


def insert_or_update_rds_record(
    connection: pymysql.Connection,
    mysql_record: Dict[str, Any],
    dry_run: bool = False
) -> Tuple[str, bool]:
    """
    Insert or update record in RDS MySQL

    Returns:
        Tuple of (action_taken, success)
        action_taken: 'inserted', 'updated', 'skipped'
    """
    if dry_run:
        return 'dry_run', True

    try:
        with connection.cursor() as cursor:
            # Check if record exists
            record_id = mysql_record.get('id')

            if record_id:
                cursor.execute(
                    "SELECT id FROM analysis_history WHERE id = %s",
                    (record_id,)
                )
                exists = cursor.fetchone()

                if exists:
                    # Update existing record
                    update_fields = []
                    update_values = []

                    for field, value in mysql_record.items():
                        if field != 'id':
                            update_fields.append(f"{field} = %s")
                            update_values.append(value)

                    update_values.append(record_id)

                    update_sql = f"""
                        UPDATE analysis_history
                        SET {', '.join(update_fields)}
                        WHERE id = %s
                    """

                    cursor.execute(update_sql, update_values)
                    connection.commit()
                    return 'updated', True

            # Insert new record
            fields = list(mysql_record.keys())
            placeholders = ', '.join(['%s'] * len(fields))

            insert_sql = f"""
                INSERT INTO analysis_history ({', '.join(fields)})
                VALUES ({placeholders})
            """

            cursor.execute(insert_sql, list(mysql_record.values()))
            connection.commit()
            return 'inserted', True

    except Exception as e:
        print(f"⚠️  Failed to insert/update record: {str(e)}")
        return 'failed', False


def rollback_migration(
    rds_conn: pymysql.Connection,
    table: Any,
    dry_run: bool = False,
    analysis_id: Optional[str] = None
) -> RollbackStats:
    """
    Rollback data from DynamoDB to RDS

    Args:
        rds_conn: RDS MySQL connection
        table: DynamoDB table
        dry_run: If True, don't actually write to RDS
        analysis_id: If provided, only rollback specific record

    Returns:
        RollbackStats with results
    """
    stats = RollbackStats()

    print("\n" + "=" * 80)
    print("DynamoDB → RDS Rollback")
    print("=" * 80)

    if dry_run:
        print("🔍 DRY RUN MODE - No data will be written to RDS")
    else:
        print("⚠️  WARNING: This will modify RDS MySQL database!")
        print("Make sure you have a backup before proceeding.")
        response = input("\nType 'YES' to confirm: ")
        if response != 'YES':
            print("Rollback cancelled.")
            return stats

    # Fetch DynamoDB records
    if analysis_id:
        print(f"\n🔍 Rolling back specific record: {analysis_id}")
        try:
            response = table.get_item(Key={'analysis_id': analysis_id})
            dynamodb_items = [response['Item']] if 'Item' in response else []
        except Exception as e:
            print(f"❌ Failed to fetch record: {str(e)}")
            return stats
    else:
        dynamodb_items = fetch_dynamodb_records(table)

    stats.total_records = len(dynamodb_items)

    if not dynamodb_items:
        print("ℹ️  No records to rollback")
        return stats

    # Transform and insert/update
    print(f"\n🚀 Starting rollback of {stats.total_records:,} records...")

    for dynamodb_item in dynamodb_items:
        try:
            # Transform to MySQL format
            mysql_record = transform_dynamodb_to_mysql(dynamodb_item)

            # Insert or update
            action, success = insert_or_update_rds_record(
                rds_conn, mysql_record, dry_run
            )

            stats.processed += 1

            if success:
                if action == 'inserted':
                    stats.inserted += 1
                elif action == 'updated':
                    stats.updated += 1
                elif action == 'dry_run':
                    stats.skipped += 1
            else:
                stats.failed += 1
                stats.errors.append({
                    'analysis_id': dynamodb_item.get('analysis_id'),
                    'mysql_id': dynamodb_item.get('mysql_id'),
                    'error': 'Insert/update failed'
                })

            # Progress report
            if stats.processed % 100 == 0:
                print(f"Processed {stats.processed:,}/{stats.total_records:,} records...")

        except Exception as e:
            stats.failed += 1
            stats.errors.append({
                'analysis_id': dynamodb_item.get('analysis_id'),
                'error': str(e)
            })

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Rollback data from DynamoDB to RDS MySQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual writes to RDS'
    )
    parser.add_argument(
        '--analysis-id',
        type=str,
        help='Rollback specific DynamoDB record by analysis_id'
    )

    args = parser.parse_args()

    # Connect to databases
    rds_conn = connect_to_rds()
    if not rds_conn:
        return 1

    dynamodb, table = connect_to_dynamodb()
    if not table:
        rds_conn.close()
        return 1

    try:
        # Run rollback
        stats = rollback_migration(rds_conn, table, args.dry_run, args.analysis_id)

        # Save log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"rollback_log_{timestamp}.json"

        log_data = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats.to_dict(),
            'errors': stats.errors
        }

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 Rollback log saved: {log_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("Rollback Summary")
        print("=" * 80)
        summary = stats.to_dict()
        print(f"Total records:     {summary['total_records']:,}")
        print(f"Processed:         {summary['processed']:,}")
        print(f"Inserted:          {summary['inserted']:,}")
        print(f"Updated:           {summary['updated']:,}")
        print(f"Skipped:           {summary['skipped']:,}")
        print(f"Failed:            {summary['failed']:,}")
        print(f"Elapsed time:      {summary['elapsed_seconds']:.2f}s")
        print("=" * 80)

        if args.dry_run:
            print("\n🔍 DRY RUN completed - No data was written to RDS")
        elif stats.failed == 0:
            print("\n✅ Rollback completed successfully!")
        else:
            print(f"\n⚠️  Rollback completed with {stats.failed} errors")

        return 0 if stats.failed == 0 else 1

    finally:
        rds_conn.close()


if __name__ == '__main__':
    sys.exit(main())
