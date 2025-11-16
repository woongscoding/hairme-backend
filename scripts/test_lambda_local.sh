#!/bin/bash
################################################################################
# Local Lambda Testing Script for HairMe Backend
#
# This script tests the Lambda container locally using Docker before deployment
# to AWS Lambda.
#
# Usage:
#   ./scripts/test_lambda_local.sh [OPTIONS]
#
# Options:
#   --port PORT           Local port for Lambda runtime (default: 9000)
#   --test-image PATH     Path to test image file (default: tests/sample_face.jpg)
#   --verbose             Show detailed Docker logs
#   --keep-running        Don't stop container after test
#
# Examples:
#   # Basic test
#   ./scripts/test_lambda_local.sh
#
#   # Test with custom image
#   ./scripts/test_lambda_local.sh --test-image ~/my_test_photo.jpg
#
#   # Run in verbose mode
#   ./scripts/test_lambda_local.sh --verbose
#
# Prerequisites:
#   - Docker installed and running
#   - Dockerfile.lambda exists
#   - Test image file available
#   - GEMINI_API_KEY environment variable set
################################################################################

set -e  # Exit on error

# ==================== Configuration ====================

# Default values
PORT=9000
TEST_IMAGE="tests/sample_face.jpg"
VERBOSE=false
KEEP_RUNNING=false
CONTAINER_NAME="hairme-lambda-test"
IMAGE_NAME="hairme-lambda-local"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== Parse Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --test-image)
            TEST_IMAGE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --keep-running)
            KEEP_RUNNING=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# ==================== Helper Functions ====================

log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

cleanup() {
    if [ "$KEEP_RUNNING" = false ]; then
        log_info "Cleaning up..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME" 2>/dev/null || true
    fi
}

# Trap to ensure cleanup on exit
trap cleanup EXIT

# ==================== Validation ====================

print_header "Local Lambda Testing"

# Check Docker
if ! command -v docker &> /dev/null; then
    log_error "Docker not installed"
    exit 1
fi

# Check Docker is running
if ! docker info &> /dev/null; then
    log_error "Docker is not running"
    exit 1
fi

# Check Dockerfile.lambda exists
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ ! -f "$PROJECT_ROOT/Dockerfile.lambda" ]; then
    log_error "Dockerfile.lambda not found in project root"
    exit 1
fi

# Check GEMINI_API_KEY
if [ -z "$GEMINI_API_KEY" ]; then
    log_warning "GEMINI_API_KEY not set - reading from .env"

    if [ -f "$PROJECT_ROOT/.env" ]; then
        export $(grep GEMINI_API_KEY "$PROJECT_ROOT/.env" | xargs)
    fi

    if [ -z "$GEMINI_API_KEY" ]; then
        log_error "GEMINI_API_KEY not found in environment or .env file"
        exit 1
    fi
fi

# Check test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    log_warning "Test image not found: $TEST_IMAGE"
    log_info "Using placeholder - some tests may fail"
fi

log_success "Prerequisites check passed"

# ==================== Build Docker Image ====================

print_header "Building Lambda Container"

cd "$PROJECT_ROOT"

log_info "Building Docker image from Dockerfile.lambda..."

if [ "$VERBOSE" = true ]; then
    docker build \
        -f Dockerfile.lambda \
        -t "$IMAGE_NAME" \
        --platform linux/amd64 \
        .
else
    docker build \
        -f Dockerfile.lambda \
        -t "$IMAGE_NAME" \
        --platform linux/amd64 \
        . > /dev/null 2>&1
fi

log_success "Docker image built: $IMAGE_NAME"

# ==================== Stop Existing Container ====================

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_info "Stopping existing container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
fi

# ==================== Run Lambda Container ====================

print_header "Starting Lambda Container"

log_info "Starting container on port $PORT..."

if [ "$VERBOSE" = true ]; then
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:8080" \
        -e GEMINI_API_KEY="$GEMINI_API_KEY" \
        -e USE_DYNAMODB=true \
        -e AWS_REGION=ap-northeast-2 \
        -e DYNAMODB_TABLE_NAME=hairme-analysis \
        -e LOG_LEVEL=DEBUG \
        "$IMAGE_NAME"
else
    docker run -d \
        --name "$CONTAINER_NAME" \
        -p "${PORT}:8080" \
        -e GEMINI_API_KEY="$GEMINI_API_KEY" \
        -e USE_DYNAMODB=true \
        -e AWS_REGION=ap-northeast-2 \
        -e DYNAMODB_TABLE_NAME=hairme-analysis \
        -e LOG_LEVEL=INFO \
        "$IMAGE_NAME" > /dev/null
fi

log_success "Container started: $CONTAINER_NAME"
log_info "Lambda runtime available at: http://localhost:${PORT}/2015-03-31/functions/function/invocations"

# Wait for container to be ready
log_info "Waiting for container to be ready..."
sleep 5

# Check container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_error "Container failed to start"
    log_info "Container logs:"
    docker logs "$CONTAINER_NAME"
    exit 1
fi

log_success "Container is running"

# ==================== Test Health Check ====================

print_header "Testing Health Check"

# Create test payload for root endpoint
HEALTH_PAYLOAD=$(cat <<EOF
{
  "resource": "/",
  "path": "/",
  "httpMethod": "GET",
  "headers": {},
  "queryStringParameters": null,
  "body": null,
  "isBase64Encoded": false
}
EOF
)

log_info "Sending health check request..."

HEALTH_RESPONSE=$(curl -s -X POST \
    "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
    -d "$HEALTH_PAYLOAD")

if echo "$HEALTH_RESPONSE" | grep -q "running"; then
    log_success "Health check passed"
    echo "$HEALTH_RESPONSE" | jq '.' 2>/dev/null || echo "$HEALTH_RESPONSE"
else
    log_error "Health check failed"
    echo "$HEALTH_RESPONSE"
    exit 1
fi

# ==================== Test API Health Endpoint ====================

print_header "Testing API Health Endpoint"

API_HEALTH_PAYLOAD=$(cat <<EOF
{
  "resource": "/api/health",
  "path": "/api/health",
  "httpMethod": "GET",
  "headers": {},
  "queryStringParameters": null,
  "body": null,
  "isBase64Encoded": false
}
EOF
)

log_info "Sending API health check request..."

API_HEALTH_RESPONSE=$(curl -s -X POST \
    "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
    -d "$API_HEALTH_PAYLOAD")

if echo "$API_HEALTH_RESPONSE" | grep -q "healthy"; then
    log_success "API health check passed"
    echo "$API_HEALTH_RESPONSE" | jq '.body | fromjson' 2>/dev/null || echo "$API_HEALTH_RESPONSE"
else
    log_error "API health check failed"
    echo "$API_HEALTH_RESPONSE"
fi

# ==================== Test Image Analysis (if test image exists) ====================

if [ -f "$TEST_IMAGE" ]; then
    print_header "Testing Image Analysis"

    log_info "Encoding test image to base64..."
    IMAGE_BASE64=$(base64 -w 0 "$TEST_IMAGE" 2>/dev/null || base64 "$TEST_IMAGE")

    ANALYZE_PAYLOAD=$(cat <<EOF
{
  "resource": "/api/analyze",
  "path": "/api/analyze",
  "httpMethod": "POST",
  "headers": {
    "Content-Type": "multipart/form-data; boundary=----WebKitFormBoundary"
  },
  "body": "$IMAGE_BASE64",
  "isBase64Encoded": true
}
EOF
)

    log_info "Sending image analysis request..."
    log_warning "This may take 10-30 seconds (Gemini API call)..."

    ANALYZE_RESPONSE=$(curl -s -X POST \
        --max-time 60 \
        "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
        -d "$ANALYZE_PAYLOAD")

    if echo "$ANALYZE_RESPONSE" | grep -q "statusCode"; then
        STATUS_CODE=$(echo "$ANALYZE_RESPONSE" | jq -r '.statusCode' 2>/dev/null || echo "unknown")

        if [ "$STATUS_CODE" = "200" ]; then
            log_success "Image analysis succeeded"

            # Extract analysis_id
            ANALYSIS_ID=$(echo "$ANALYZE_RESPONSE" | jq -r '.body | fromjson | .analysis_id' 2>/dev/null)

            if [ -n "$ANALYSIS_ID" ] && [ "$ANALYSIS_ID" != "null" ]; then
                log_success "Analysis ID: $ANALYSIS_ID"

                # Save for feedback test
                echo "$ANALYSIS_ID" > /tmp/hairme_test_analysis_id.txt
            fi

            echo "$ANALYZE_RESPONSE" | jq '.body | fromjson' 2>/dev/null || echo "$ANALYZE_RESPONSE"
        else
            log_warning "Analysis returned status code: $STATUS_CODE"
            echo "$ANALYZE_RESPONSE" | jq '.' 2>/dev/null || echo "$ANALYZE_RESPONSE"
        fi
    else
        log_error "Analysis request failed"
        echo "$ANALYZE_RESPONSE"
    fi
else
    log_warning "Skipping image analysis test (no test image)"
fi

# ==================== Test Feedback Endpoint ====================

if [ -f /tmp/hairme_test_analysis_id.txt ]; then
    print_header "Testing Feedback Endpoint"

    ANALYSIS_ID=$(cat /tmp/hairme_test_analysis_id.txt)

    FEEDBACK_BODY=$(cat <<EOF
{
  "analysis_id": "$ANALYSIS_ID",
  "style_index": 1,
  "feedback": "good",
  "naver_clicked": true
}
EOF
)

    FEEDBACK_PAYLOAD=$(cat <<EOF
{
  "resource": "/api/feedback",
  "path": "/api/feedback",
  "httpMethod": "POST",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": $(echo "$FEEDBACK_BODY" | jq -c '.'),
  "isBase64Encoded": false
}
EOF
)

    log_info "Sending feedback request for analysis: $ANALYSIS_ID"

    FEEDBACK_RESPONSE=$(curl -s -X POST \
        "http://localhost:${PORT}/2015-03-31/functions/function/invocations" \
        -d "$FEEDBACK_PAYLOAD")

    if echo "$FEEDBACK_RESPONSE" | grep -q '"message".*"success"'; then
        log_success "Feedback submission succeeded"
        echo "$FEEDBACK_RESPONSE" | jq '.body | fromjson' 2>/dev/null || echo "$FEEDBACK_RESPONSE"
    else
        log_warning "Feedback submission may have failed"
        echo "$FEEDBACK_RESPONSE" | jq '.' 2>/dev/null || echo "$FEEDBACK_RESPONSE"
    fi

    # Cleanup temp file
    rm -f /tmp/hairme_test_analysis_id.txt
fi

# ==================== Show Container Logs ====================

if [ "$VERBOSE" = true ]; then
    print_header "Container Logs"
    docker logs "$CONTAINER_NAME"
fi

# ==================== Summary ====================

print_header "Test Summary"

log_success "Local Lambda testing completed!"
echo ""
echo "Container Details:"
echo "  Name:        $CONTAINER_NAME"
echo "  Image:       $IMAGE_NAME"
echo "  Port:        $PORT"
echo "  Status:      $(docker ps --filter name=$CONTAINER_NAME --format '{{.Status}}')"
echo ""

if [ "$KEEP_RUNNING" = true ]; then
    log_info "Container is still running (--keep-running flag)"
    echo ""
    echo "Useful Commands:"
    echo "  View logs:    docker logs $CONTAINER_NAME -f"
    echo "  Stop:         docker stop $CONTAINER_NAME"
    echo "  Remove:       docker rm $CONTAINER_NAME"
    echo "  Invoke:       curl -X POST http://localhost:${PORT}/2015-03-31/functions/function/invocations -d '{...}'"
else
    log_info "Container will be stopped and removed"
fi

echo ""
echo "Next Steps:"
echo "  1. If tests passed, deploy to AWS: ./scripts/deploy_lambda.sh"
echo "  2. Monitor Lambda logs: aws logs tail /aws/lambda/hairme-analyze --follow"
echo "  3. Test deployed Lambda: aws lambda invoke --function-name hairme-analyze output.json"

print_header "Testing Complete"
