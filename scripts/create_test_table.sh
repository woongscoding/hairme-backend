#!/bin/bash
################################################################################
# HairMe DynamoDB *Test* Table Creation Script
#
# Purpose: tests/test_dynamodb_integration.py 전용 테이블을 만든다.
#          운영 테이블(hairme-analysis)과 동일한 키/GSI 스키마를 쓰되,
#          테스트 레코드가 스스로 사라지도록 TTL 속성을 활성화한다.
#
# 배경:
#   통합 테스트는 실제 DynamoDB 에 쓰기를 수행한다. 예전에는 자격증명만
#   있으면 기본 테이블 이름(hairme-analysis = 운영)으로 실행되어 운영
#   테이블에 테스트 레코드가 쌓였다. 지금은 아래 두 변수를 모두 지정해야만
#   실행되며, 운영 테이블 이름을 지정하면 테스트가 거부된다.
#
#     RUN_DYNAMODB_INTEGRATION=1
#     DYNAMODB_TEST_TABLE_NAME=hairme-analysis-test
#
# Usage:
#   chmod +x scripts/create_test_table.sh
#   ./scripts/create_test_table.sh [region]
#
# Arguments:
#   region (optional): AWS region (default: ap-northeast-2)
#
# Prerequisites:
#   - AWS CLI installed and configured (aws configure)
#   - IAM permissions: dynamodb:CreateTable, dynamodb:DescribeTable,
#                      dynamodb:UpdateTimeToLive
#
# Table Schema (운영 테이블 describe-table 결과와 동일):
#   - Primary Key: analysis_id (String, HASH only - range key 없음)
#   - GSI: created_at-index (entity_type HASH + created_at RANGE, Projection ALL)
#   - Billing: PAY_PER_REQUEST
#   - TTL: 속성명 "ttl" (테스트 테이블에만 적용 - 운영은 DISABLED)
#
# 주의:
#   PITR 은 켜지 않는다 (테스트 데이터에 불필요한 비용).
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
REGION="${1:-ap-northeast-2}"
TABLE_NAME="hairme-analysis-test"
PRODUCTION_TABLE_NAME="hairme-analysis"
TTL_ATTRIBUTE="ttl"

# 안전장치: 이 스크립트는 절대 운영 테이블을 건드리지 않는다
if [ "$TABLE_NAME" = "$PRODUCTION_TABLE_NAME" ]; then
    echo "❌ Error: 테스트 테이블 이름이 운영 테이블과 같습니다"
    exit 1
fi

echo "========================================="
echo "HairMe DynamoDB TEST Table Creation"
echo "========================================="
echo "Region: $REGION"
echo "Table:  $TABLE_NAME  (운영 테이블 $PRODUCTION_TABLE_NAME 아님)"
echo ""

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ Error: AWS CLI is not installed"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    echo "❌ Error: AWS credentials not configured"
    echo "Run: aws configure"
    exit 1
fi

echo "✅ AWS CLI configured"
echo ""

# Check if table already exists
echo "Checking if test table already exists..."
if aws dynamodb describe-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" &> /dev/null; then
    echo "✅ Table '$TABLE_NAME' already exists in region $REGION - nothing to do"
    aws dynamodb describe-table \
        --table-name "$TABLE_NAME" \
        --region "$REGION" \
        --query 'Table.[TableName,TableStatus,ItemCount]' \
        --output table
    exit 0
fi

echo "✅ Table does not exist, proceeding with creation..."
echo ""
echo "Creating DynamoDB test table..."
echo "This may take 1-2 minutes..."
echo ""

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
        Key=Environment,Value=Test \
        Key=Purpose,Value=IntegrationTests \
        Key=CostCenter,Value=Backend \
    --region "$REGION" \
    --output json

echo ""
echo "⏳ Waiting for table to become ACTIVE..."
aws dynamodb wait table-exists \
    --table-name "$TABLE_NAME" \
    --region "$REGION"

echo ""
echo "✅ Test table created successfully!"
echo ""

# TTL 활성화 - 남겨진 테스트 레코드가 스스로 만료되도록
echo "Enabling TTL on attribute '$TTL_ATTRIBUTE'..."
aws dynamodb update-time-to-live \
    --table-name "$TABLE_NAME" \
    --time-to-live-specification "Enabled=true,AttributeName=$TTL_ATTRIBUTE" \
    --region "$REGION" \
    --output json > /dev/null

echo "✅ TTL enabled (속성: $TTL_ATTRIBUTE)"
echo ""

# Display table details
echo "========================================="
echo "Test Table Details"
echo "========================================="
aws dynamodb describe-table \
    --table-name "$TABLE_NAME" \
    --region "$REGION" \
    --query 'Table.{Name:TableName,Status:TableStatus,Billing:BillingModeSummary.BillingMode,GSI:GlobalSecondaryIndexes[0].IndexName}' \
    --output table

aws dynamodb describe-time-to-live \
    --table-name "$TABLE_NAME" \
    --region "$REGION" \
    --output table

echo ""
echo "========================================="
echo "Next Steps"
echo "========================================="
echo "통합 테스트 실행:"
echo ""
echo "  RUN_DYNAMODB_INTEGRATION=1 \\"
echo "  DYNAMODB_TEST_TABLE_NAME=$TABLE_NAME \\"
echo "  pytest tests/test_dynamodb_integration.py -v"
echo ""
echo "두 변수 중 하나라도 없으면 스위트는 전부 skip 된다."
echo "========================================="
