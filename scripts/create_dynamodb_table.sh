#!/bin/bash
################################################################################
# HairMe DynamoDB Table Creation Script
#
# Purpose: Create the hairme-analysis DynamoDB table using AWS CLI
# Source: Migrated from RDS MySQL analysis_history table
#
# Usage:
#   chmod +x scripts/create_dynamodb_table.sh
#   ./scripts/create_dynamodb_table.sh [region]
#
# Arguments:
#   region (optional): AWS region (default: ap-northeast-2)
#
# Examples:
#   ./scripts/create_dynamodb_table.sh                    # Use default region
#   ./scripts/create_dynamodb_table.sh us-east-1          # Use specific region
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - IAM permissions: dynamodb:CreateTable, dynamodb:DescribeTable
#
# Table Schema:
#   - Primary Key: analysis_id (String, UUID)
#   - GSI: created_at-index (entity_type + created_at for time-based queries)
#   - Billing: PAY_PER_REQUEST (on-demand, no provisioning needed)
#
# Estimated Cost:
#   - Free Tier: 25 GB storage + 25 WCU/RCU per month
#   - Pay-per-request: $1.25 per million write requests
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
REGION="${1:-ap-northeast-2}"
TABLE_NAME="hairme-analysis"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "HairMe DynamoDB Table Creation"
echo "========================================="
echo "Region: $REGION"
echo "Table: $TABLE_NAME"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ Error: AWS CLI is not installed"
    echo "Install: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ Error: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✅ AWS CLI configured"
CALLER_IDENTITY=$(aws sts get-caller-identity --output json)
ACCOUNT_ID=$(echo "$CALLER_IDENTITY" | grep -o '"Account": "[^"]*"' | cut -d'"' -f4)
echo "Account ID: $ACCOUNT_ID"
echo ""

# Check if table already exists
echo "Checking if table already exists..."
if aws dynamodb describe-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" &> /dev/null; then
    echo "⚠️  Table '$TABLE_NAME' already exists in region $REGION"
    echo ""
    echo "Table details:"
    aws dynamodb describe-table \
        --table-name "$TABLE_NAME" \
        --region "$REGION" \
        --query 'Table.[TableName,TableStatus,ItemCount,TableSizeBytes,CreationDateTime]' \
        --output table
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
else
    echo "✅ Table does not exist, proceeding with creation..."
fi

echo ""
echo "Creating DynamoDB table..."
echo "This may take 1-2 minutes..."
echo ""

# Create table using AWS CLI
aws dynamodb create-table \
    --table-name "$TABLE_NAME" \
    --attribute-definitions \
        AttributeName=analysis_id,AttributeType=S \
        AttributeName=entity_type,AttributeType=S \
        AttributeName=created_at,AttributeType=S \
    --key-schema \
        AttributeName=analysis_id,KeyType=HASH \
    --global-secondary-indexes \
        "IndexName=created_at-index,KeySchema=[{AttributeName=entity_type,KeyType=HASH},{AttributeName=created_at,KeyType=RANGE}],Projection={ProjectionType=ALL}" \
    --billing-mode PAY_PER_REQUEST \
    --tags \
        Key=Project,Value=HairMe \
        Key=Environment,Value=Production \
        Key=MigrationSource,Value=RDS-MySQL \
        Key=CostCenter,Value=Backend \
    --region "$REGION" \
    --output json

echo ""
echo "⏳ Waiting for table to become ACTIVE..."

# Wait for table to be active
aws dynamodb wait table-exists \
    --table-name "$TABLE_NAME" \
    --region "$REGION"

echo ""
echo "✅ Table created successfully!"
echo ""

# Enable Point-in-Time Recovery
echo "Enabling Point-in-Time Recovery (PITR)..."
aws dynamodb update-continuous-backups \
    --table-name "$TABLE_NAME" \
    --point-in-time-recovery-specification PointInTimeRecoveryEnabled=true \
    --region "$REGION" \
    --output json > /dev/null

echo "✅ PITR enabled"
echo ""

# Display table details
echo "========================================="
echo "Table Details"
echo "========================================="
aws dynamodb describe-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" \
    --query 'Table.{Name:TableName,Status:TableStatus,Billing:BillingModeSummary.BillingMode,GSI:GlobalSecondaryIndexes[0].IndexName,PITR:PointInTimeRecoveryDescription.PointInTimeRecoveryStatus}' \
    --output table

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo "1. Test connection:"
echo "   python scripts/test_dynamodb_connection.py"
echo ""
echo "2. Migrate data from RDS:"
echo "   python scripts/migrate_rds_to_dynamodb.py"
echo ""
echo "3. Update application code to use DynamoDB:"
echo "   - Update database/dynamodb_client.py"
echo "   - Update api/endpoints/analyze.py"
echo "   - Update api/endpoints/feedback.py"
echo ""
echo "========================================="
echo "Table ARN:"
TABLE_ARN=$(aws dynamodb describe-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" \
    --query 'Table.TableArn' \
    --output text)
echo "$TABLE_ARN"
echo "========================================="
