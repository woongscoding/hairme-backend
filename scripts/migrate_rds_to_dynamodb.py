#!/usr/bin/env python3
"""
RDS MySQL to DynamoDB Migration Script

Migrates all data from analysis_history table in RDS MySQL to DynamoDB
with data validation, batch processing, and error handling.

Usage:
    # Dry-run mode (no actual writes)
    python scripts/migrate_rds_to_dynamodb.py --dry-run

    # Actual migration
    python scripts/migrate_rds_to_dynamodb.py

    # With custom batch size
    python scripts/migrate_rds_to_dynamodb.py --batch-size 10

    # Resume from specific ID
    python scripts/migrate_rds_to_dynamodb.py --start-from 1000

Prerequisites:
    - RDS MySQL accessible (DATABASE_URL, DB_PASSWORD in .env)
    - DynamoDB table created (hairme-analysis)
    - AWS credentials configured
    - pip install pymysql boto3 python-dotenv tqdm

Output:
    - migration_log_YYYYMMDD_HHMMSS.json (detailed log)
    - migration_errors_YYYYMMDD_HHMMSS.json (failed records)
"""

import sys
import os
import json
import argparse
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import io

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pymysql
    from pymysql.cursors import DictCursor
except ImportError:
    print("❌ Error: pymysql not installed. Run: pip install pymysql")
    sys.exit(1)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ Error: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("❌ Error: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠️  Warning: tqdm not installed. Progress bar disabled.")
    print("Install: pip install tqdm")
    tqdm = None

# Load environment variables from project root (.env overrides system env vars)
env_path = project_root / '.env'
load_dotenv(env_path, override=True)


class MigrationStats:
    """Track migration statistics"""

    def __init__(self):
        self.total_records = 0
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.start_time = datetime.now()
        self.errors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            'total_records': self.total_records,
            'processed': self.processed,
            'successful': self.successful,
            'failed': self.failed,
            'skipped': self.skipped,
            'elapsed_seconds': round(elapsed, 2),
            'records_per_second': round(self.successful / elapsed, 2) if elapsed > 0 else 0,
            'success_rate': round(self.successful / self.processed * 100, 2) if self.processed > 0 else 0
        }


def convert_to_decimal(value: Any) -> Any:
    """Convert Python types to DynamoDB-compatible types"""
    if isinstance(value, float):
        return Decimal(str(value))
    elif isinstance(value, dict):
        return {k: convert_to_decimal(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_decimal(item) for item in value]
    return value


def transform_mysql_to_dynamodb(mysql_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform MySQL record to DynamoDB item format

    Key transformations:
    - id (INT) → analysis_id (UUID string)
    - created_at (DATETIME) → created_at (ISO 8601 string)
    - Float → Decimal
    - NULL → omit from item
    """
    # Generate UUID for DynamoDB (preserving MySQL ID in metadata)
    analysis_id = str(uuid.uuid4())

    # Build DynamoDB item
    item = {
        # Primary Key & GSI
        'analysis_id': analysis_id,
        'entity_type': 'ANALYSIS',  # For GSI
        'created_at': mysql_record['created_at'].isoformat() if mysql_record.get('created_at') else datetime.now(timezone.utc).isoformat(),

        # Store original MySQL ID for reference
        'mysql_id': mysql_record['id'],

        # Basic Information
        'user_id': mysql_record.get('user_id') or 'anonymous',
        'image_hash': mysql_record.get('image_hash') or '',
        'face_shape': mysql_record.get('face_shape'),
        'personal_color': mysql_record.get('personal_color'),
        'processing_time': mysql_record.get('processing_time'),
        'detection_method': mysql_record.get('detection_method') or 'unknown',
    }

    # Handle JSON columns
    if mysql_record.get('recommendations'):
        if isinstance(mysql_record['recommendations'], str):
            try:
                item['recommendations'] = json.loads(mysql_record['recommendations'])
            except json.JSONDecodeError:
                item['recommendations'] = []
        else:
            item['recommendations'] = mysql_record['recommendations']
    else:
        item['recommendations'] = []

    if mysql_record.get('recommended_styles'):
        if isinstance(mysql_record['recommended_styles'], str):
            try:
                item['recommended_styles'] = json.loads(mysql_record['recommended_styles'])
            except json.JSONDecodeError:
                item['recommended_styles'] = []
        else:
            item['recommended_styles'] = mysql_record['recommended_styles']
    else:
        item['recommended_styles'] = []

    # OpenCV measurements (10 fields)
    opencv_fields = [
        'opencv_face_ratio', 'opencv_forehead_ratio', 'opencv_cheekbone_ratio',
        'opencv_jaw_ratio', 'opencv_prediction', 'opencv_confidence',
        'opencv_gemini_agreement', 'opencv_upper_face_ratio',
        'opencv_middle_face_ratio', 'opencv_lower_face_ratio'
    ]
    for field in opencv_fields:
        if mysql_record.get(field) is not None:
            item[field] = mysql_record[field]

    # MediaPipe measurements (10 fields)
    mediapipe_fields = [
        'mediapipe_face_ratio', 'mediapipe_forehead_width',
        'mediapipe_cheekbone_width', 'mediapipe_jaw_width',
        'mediapipe_forehead_ratio', 'mediapipe_jaw_ratio',
        'mediapipe_ITA_value', 'mediapipe_hue_value',
        'mediapipe_confidence', 'mediapipe_features_complete'
    ]
    for field in mediapipe_fields:
        if mysql_record.get(field) is not None:
            item[field] = mysql_record[field]

    # Feedback data (7 fields)
    feedback_fields = [
        'style_1_feedback', 'style_2_feedback', 'style_3_feedback',
        'style_1_naver_clicked', 'style_2_naver_clicked', 'style_3_naver_clicked'
    ]
    for field in feedback_fields:
        value = mysql_record.get(field)
        if value is not None:
            item[field] = value
        else:
            # Set defaults for feedback fields
            if 'feedback' in field:
                item[field] = None
            elif 'clicked' in field:
                item[field] = False

    # Feedback timestamp
    if mysql_record.get('feedback_at'):
        item['feedback_at'] = mysql_record['feedback_at'].isoformat()
    else:
        item['feedback_at'] = None

    # Convert floats to Decimal for DynamoDB
    item = convert_to_decimal(item)

    return item


def connect_to_rds() -> Optional[pymysql.Connection]:
    """Connect to RDS MySQL database"""
    database_url = os.getenv('DATABASE_URL')
    db_password = os.getenv('DB_PASSWORD')

    print(f"DEBUG: DATABASE_URL = {database_url}")
    print(f"DEBUG: DB_PASSWORD = {'***' if db_password else 'NOT SET'}")

    if not database_url:
        print("❌ Error: DATABASE_URL not set in .env")
        return None

    try:
        # Parse connection string
        # Format: mysql+pymysql://admin@hostname:port/database
        import urllib.parse

        if "://admin@" in database_url and db_password:
            # Inject password
            encoded_password = urllib.parse.quote_plus(db_password)
            db_url = database_url.replace("asyncmy", "pymysql").replace(
                "://admin@",
                f"://admin:{encoded_password}@"
            )
        else:
            db_url = database_url.replace("asyncmy", "pymysql")

        # Extract connection parameters
        from sqlalchemy.engine.url import make_url
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

        # Verify table exists
        table.load()

        print(f"✅ Connected to DynamoDB: {table_name} ({aws_region})")
        return dynamodb, table

    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print(f"❌ DynamoDB table '{table_name}' not found")
            print(f"Create it: ./scripts/create_dynamodb_table.sh")
        else:
            print(f"❌ DynamoDB connection failed: {e.response['Error']['Message']}")
        return None, None
    except Exception as e:
        print(f"❌ Failed to connect to DynamoDB: {str(e)}")
        return None, None


def fetch_mysql_records(
    connection: pymysql.Connection,
    start_from: int = 0,
    batch_size: int = 100
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch all records from MySQL in batches

    Returns:
        Tuple of (all_records, total_count)
    """
    print(f"\n📊 Fetching records from MySQL (starting from ID {start_from})...")

    try:
        with connection.cursor() as cursor:
            # Get total count
            cursor.execute("SELECT COUNT(*) as total FROM analysis_history WHERE id >= %s", (start_from,))
            total = cursor.fetchone()['total']

            print(f"Total records to migrate: {total:,}")

            if total == 0:
                return [], 0

            # Fetch all records
            cursor.execute("""
                SELECT * FROM analysis_history
                WHERE id >= %s
                ORDER BY id ASC
            """, (start_from,))

            records = cursor.fetchall()
            print(f"✅ Fetched {len(records):,} records from MySQL")

            return records, total

    except Exception as e:
        print(f"❌ Failed to fetch records: {str(e)}")
        return [], 0


def migrate_batch_to_dynamodb(
    table: Any,
    items: List[Dict[str, Any]],
    dry_run: bool = False
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Write a batch of items to DynamoDB

    Args:
        table: DynamoDB table resource
        items: List of DynamoDB items (max 25)
        dry_run: If True, skip actual write

    Returns:
        Tuple of (success_count, failed_items)
    """
    if dry_run:
        return len(items), []

    try:
        with table.batch_writer() as batch:
            for item in items:
                batch.put_item(Item=item)

        return len(items), []

    except ClientError as e:
        print(f"⚠️  Batch write failed: {e.response['Error']['Message']}")
        # Return all items as failed
        return 0, items
    except Exception as e:
        print(f"⚠️  Unexpected error in batch write: {str(e)}")
        return 0, items


def save_migration_log(stats: MigrationStats, output_file: str):
    """Save migration statistics to JSON file"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'statistics': stats.to_dict(),
        'errors': stats.errors
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Migration log saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate data from RDS MySQL to DynamoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Perform dry run without actual writes to DynamoDB'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=25,
        help='Batch size for DynamoDB writes (max 25, default: 25)'
    )
    parser.add_argument(
        '--start-from',
        type=int,
        default=0,
        help='Start migration from specific MySQL ID (for resuming)'
    )

    args = parser.parse_args()

    # Validate batch size
    if args.batch_size > 25:
        print("⚠️  Warning: DynamoDB batch_write max is 25 items. Using 25.")
        args.batch_size = 25

    print("=" * 80)
    print("RDS MySQL → DynamoDB Migration")
    print("=" * 80)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No data will be written to DynamoDB")

    print(f"Batch size: {args.batch_size}")
    print(f"Start from ID: {args.start_from}")
    print()

    # Initialize statistics
    stats = MigrationStats()

    # Connect to databases
    rds_conn = connect_to_rds()
    if not rds_conn:
        print("❌ Migration aborted: Cannot connect to RDS")
        return 1

    dynamodb, table = connect_to_dynamodb()
    if not table and not args.dry_run:
        print("❌ Migration aborted: Cannot connect to DynamoDB")
        rds_conn.close()
        return 1

    try:
        # Fetch all records from MySQL
        mysql_records, total_count = fetch_mysql_records(rds_conn, args.start_from)

        if not mysql_records:
            print("ℹ️  No records to migrate")
            rds_conn.close()
            return 0

        stats.total_records = total_count

        # Transform and migrate in batches
        print(f"\n🚀 Starting migration...")
        print(f"Processing {len(mysql_records):,} records in batches of {args.batch_size}")
        print()

        batch = []
        batch_num = 0

        # Use tqdm if available
        iterator = tqdm(mysql_records, desc="Migrating") if tqdm else mysql_records

        for mysql_record in iterator:
            try:
                # Transform record
                dynamodb_item = transform_mysql_to_dynamodb(mysql_record)
                batch.append(dynamodb_item)
                stats.processed += 1

                # Write batch when full
                if len(batch) >= args.batch_size:
                    batch_num += 1
                    success_count, failed_items = migrate_batch_to_dynamodb(
                        table, batch, args.dry_run
                    )

                    stats.successful += success_count
                    stats.failed += len(failed_items)

                    # Log failures
                    for item in failed_items:
                        stats.errors.append({
                            'mysql_id': item.get('mysql_id'),
                            'analysis_id': item.get('analysis_id'),
                            'error': 'Batch write failed'
                        })

                    if not tqdm:
                        print(f"Batch {batch_num}: {success_count}/{len(batch)} succeeded")

                    batch = []

            except Exception as e:
                stats.failed += 1
                stats.errors.append({
                    'mysql_id': mysql_record.get('id'),
                    'error': str(e)
                })
                print(f"\n⚠️  Failed to transform record ID {mysql_record.get('id')}: {str(e)}")

        # Write remaining batch
        if batch:
            batch_num += 1
            success_count, failed_items = migrate_batch_to_dynamodb(
                table, batch, args.dry_run
            )

            stats.successful += success_count
            stats.failed += len(failed_items)

            for item in failed_items:
                stats.errors.append({
                    'mysql_id': item.get('mysql_id'),
                    'analysis_id': item.get('analysis_id'),
                    'error': 'Batch write failed'
                })

            if not tqdm:
                print(f"Batch {batch_num}: {success_count}/{len(batch)} succeeded")

        # Save migration log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"migration_log_{timestamp}.json"
        save_migration_log(stats, log_file)

        # Save errors if any
        if stats.errors:
            error_file = f"migration_errors_{timestamp}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(stats.errors, f, indent=2, ensure_ascii=False)
            print(f"❌ Error log saved: {error_file}")

        # Print summary
        print("\n" + "=" * 80)
        print("Migration Summary")
        print("=" * 80)
        summary = stats.to_dict()
        print(f"Total records:     {summary['total_records']:,}")
        print(f"Processed:         {summary['processed']:,}")
        print(f"Successful:        {summary['successful']:,}")
        print(f"Failed:            {summary['failed']:,}")
        print(f"Success rate:      {summary['success_rate']:.2f}%")
        print(f"Elapsed time:      {summary['elapsed_seconds']:.2f}s")
        print(f"Records/second:    {summary['records_per_second']:.2f}")
        print("=" * 80)

        if args.dry_run:
            print("\n🔍 DRY RUN completed - No data was written to DynamoDB")
        elif stats.failed == 0:
            print("\n✅ Migration completed successfully!")
        else:
            print(f"\n⚠️  Migration completed with {stats.failed} errors")
            print(f"Check {error_file} for details")

        return 0 if stats.failed == 0 else 1

    finally:
        rds_conn.close()


if __name__ == '__main__':
    sys.exit(main())
