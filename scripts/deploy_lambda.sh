#!/bin/bash
################################################################################
# AWS Lambda Deployment Script for HairMe Backend
#
# This script automates the deployment of HairMe FastAPI application to AWS Lambda
# using Docker containers and Amazon ECR.
#
# Usage:
#   ./scripts/deploy_lambda.sh [OPTIONS]
#
# Options:
#   --function-name NAME    Lambda function name (default: hairme-analyze)
#   --region REGION         AWS region (default: ap-northeast-2)
#   --memory MB             Memory allocation (default: 2048)
#   --timeout SECONDS       Timeout in seconds (default: 30)
#   --no-backup            Skip backup of previous version
#   --dry-run              Show what would be deployed without executing
#
# Examples:
#   # Deploy with defaults
#   ./scripts/deploy_lambda.sh
#
#   # Deploy with custom settings
#   ./scripts/deploy_lambda.sh --function-name hairme-prod --memory 4096
#
#   # Dry run
#   ./scripts/deploy_lambda.sh --dry-run
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - Docker installed and running
#   - IAM permissions for ECR, Lambda, IAM
#   - Lambda function already created (or use --create flag)
#
# Environment Variables Required:
#   GEMINI_API_KEY          Gemini API key
#   USE_DYNAMODB           Set to 'true' for DynamoDB
#   AWS_REGION             AWS region
#   DYNAMODB_TABLE_NAME    DynamoDB table name
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==================== Configuration ====================

# Default values
FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-hairme-analyze}"
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
MEMORY_SIZE=1536
TIMEOUT=30
NO_BACKUP=false
DRY_RUN=false
CREATE_FUNCTION=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== Parse Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --function-name)
            FUNCTION_NAME="$2"
            shift 2
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --memory)
            MEMORY_SIZE="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --no-backup)
            NO_BACKUP=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --create)
            CREATE_FUNCTION=true
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

# ==================== Validation ====================

print_header "AWS Lambda Deployment"

log_info "Function: $FUNCTION_NAME"
log_info "Region: $AWS_REGION"
log_info "Memory: ${MEMORY_SIZE}MB"
log_info "Timeout: ${TIMEOUT}s"

if [ "$DRY_RUN" = true ]; then
    log_warning "DRY RUN MODE - No actual deployment"
fi

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not installed"
    exit 1
fi

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

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    log_error "AWS credentials not configured"
    log_info "Run: aws configure"
    exit 1
fi

log_success "Prerequisites check passed"

# Get AWS Account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "AWS Account: $ACCOUNT_ID"

# ==================== ECR Configuration ====================

print_header "ECR Configuration"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_REPOSITORY="hairme-lambda"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}:${IMAGE_TAG}"

log_info "ECR Registry: $ECR_REGISTRY"
log_info "Repository: $ECR_REPOSITORY"
log_info "Image Tag: $IMAGE_TAG"

# Check if ECR repository exists
if ! aws ecr describe-repositories \
    --repository-names "$ECR_REPOSITORY" \
    --region "$AWS_REGION" &> /dev/null; then

    log_warning "ECR repository '$ECR_REPOSITORY' not found"

    if [ "$DRY_RUN" = false ]; then
        log_info "Creating ECR repository..."
        aws ecr create-repository \
            --repository-name "$ECR_REPOSITORY" \
            --region "$AWS_REGION" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
        log_success "ECR repository created"
    fi
else
    log_success "ECR repository exists"
fi

# ==================== Docker Login to ECR ====================

print_header "Docker Authentication"

if [ "$DRY_RUN" = false ]; then
    log_info "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | \
        docker login --username AWS --password-stdin "$ECR_REGISTRY"
    log_success "Logged in to ECR"
fi

# ==================== Backup Current Version ====================

if [ "$NO_BACKUP" = false ] && [ "$DRY_RUN" = false ]; then
    print_header "Backup Current Version"

    # Check if Lambda function exists
    if aws lambda get-function \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION" &> /dev/null; then

        # Get current image URI
        CURRENT_IMAGE=$(aws lambda get-function \
            --function-name "$FUNCTION_NAME" \
            --region "$AWS_REGION" \
            --query 'Code.ImageUri' \
            --output text)

        log_info "Current image: $CURRENT_IMAGE"

        # Tag current image as backup
        BACKUP_TAG="backup-$(date +%Y%m%d-%H%M%S)"

        log_info "Creating backup tag: $BACKUP_TAG"

        # Pull current image
        docker pull "$CURRENT_IMAGE" || log_warning "Failed to pull current image"

        # Tag as backup
        docker tag "$CURRENT_IMAGE" "${ECR_REGISTRY}/${ECR_REPOSITORY}:${BACKUP_TAG}" || true

        # Push backup
        docker push "${ECR_REGISTRY}/${ECR_REPOSITORY}:${BACKUP_TAG}" || log_warning "Failed to push backup"

        log_success "Backup created: $BACKUP_TAG"
    else
        log_info "Function doesn't exist yet - skipping backup"
    fi
fi

# ==================== Build Docker Image ====================

print_header "Building Docker Image"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

log_info "Building image from Dockerfile.lambda..."

if [ "$DRY_RUN" = false ]; then
    docker build \
        -f Dockerfile.lambda \
        -t "$ECR_REPOSITORY:$IMAGE_TAG" \
        -t "$FULL_IMAGE_NAME" \
        --platform linux/amd64 \
        .

    log_success "Docker image built"
else
    log_info "Would build: docker build -f Dockerfile.lambda -t $FULL_IMAGE_NAME ."
fi

# ==================== Push to ECR ====================

print_header "Pushing to ECR"

if [ "$DRY_RUN" = false ]; then
    log_info "Pushing image to ECR..."
    docker push "$FULL_IMAGE_NAME"
    log_success "Image pushed to ECR"

    # Get image digest
    IMAGE_DIGEST=$(aws ecr describe-images \
        --repository-name "$ECR_REPOSITORY" \
        --image-ids imageTag="$IMAGE_TAG" \
        --region "$AWS_REGION" \
        --query 'imageDetails[0].imageDigest' \
        --output text)

    log_info "Image digest: $IMAGE_DIGEST"
else
    log_info "Would push: docker push $FULL_IMAGE_NAME"
fi

# ==================== Update Lambda Function ====================

print_header "Updating Lambda Function"

# Check if function exists
if aws lambda get-function \
    --function-name "$FUNCTION_NAME" \
    --region "$AWS_REGION" &> /dev/null; then

    log_info "Updating existing function: $FUNCTION_NAME"

    if [ "$DRY_RUN" = false ]; then
        # Update function code
        aws lambda update-function-code \
            --function-name "$FUNCTION_NAME" \
            --image-uri "$FULL_IMAGE_NAME" \
            --region "$AWS_REGION" \
            --output json > /dev/null

        log_success "Function code updated"

        # Wait for update to complete
        log_info "Waiting for update to complete..."
        aws lambda wait function-updated \
            --function-name "$FUNCTION_NAME" \
            --region "$AWS_REGION"

        # Update function configuration
        log_info "Updating function configuration..."
        aws lambda update-function-configuration \
            --function-name "$FUNCTION_NAME" \
            --memory-size "$MEMORY_SIZE" \
            --timeout "$TIMEOUT" \
            --environment "Variables={
                USE_DYNAMODB=true,
                AWS_REGION=$AWS_REGION,
                DYNAMODB_TABLE_NAME=hairme-analysis,
                GEMINI_API_KEY=${GEMINI_API_KEY:-},
                LOG_LEVEL=INFO,
                MODEL_NAME=gemini-1.5-flash-latest
            }" \
            --region "$AWS_REGION" \
            --output json > /dev/null

        log_success "Function configuration updated"
    else
        log_info "Would update function code"
        log_info "Would update function configuration"
    fi

elif [ "$CREATE_FUNCTION" = true ]; then
    log_info "Creating new function: $FUNCTION_NAME"

    if [ "$DRY_RUN" = false ]; then
        # Get or create execution role
        ROLE_NAME="hairme-lambda-role"
        ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || echo "")

        if [ -z "$ROLE_ARN" ]; then
            log_info "Creating IAM role..."

            # Create trust policy
            cat > /tmp/trust-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

            aws iam create-role \
                --role-name "$ROLE_NAME" \
                --assume-role-policy-document file:///tmp/trust-policy.json

            # Attach policies
            aws iam attach-role-policy \
                --role-name "$ROLE_NAME" \
                --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

            # Attach custom DynamoDB policy
            aws iam put-role-policy \
                --role-name "$ROLE_NAME" \
                --policy-name hairme-dynamodb-policy \
                --policy-document file://infrastructure/lambda_iam_policy.json

            ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text)

            log_success "IAM role created: $ROLE_ARN"

            # Wait for role to be ready
            log_info "Waiting for role to propagate..."
            sleep 10
        fi

        # Create Lambda function
        aws lambda create-function \
            --function-name "$FUNCTION_NAME" \
            --package-type Image \
            --code ImageUri="$FULL_IMAGE_NAME" \
            --role "$ROLE_ARN" \
            --memory-size "$MEMORY_SIZE" \
            --timeout "$TIMEOUT" \
            --environment "Variables={
                USE_DYNAMODB=true,
                AWS_REGION=$AWS_REGION,
                DYNAMODB_TABLE_NAME=hairme-analysis,
                GEMINI_API_KEY=${GEMINI_API_KEY:-},
                LOG_LEVEL=INFO,
                MODEL_NAME=gemini-1.5-flash-latest
            }" \

            --region "$AWS_REGION"

        log_success "Lambda function created"
    else
        log_info "Would create new function"
    fi
else
    log_error "Function '$FUNCTION_NAME' not found"
    log_info "Use --create flag to create a new function"
    exit 1
fi

# ==================== Summary ====================

print_header "Deployment Summary"

if [ "$DRY_RUN" = false ]; then
    # Get function info
    FUNCTION_INFO=$(aws lambda get-function \
        --function-name "$FUNCTION_NAME" \
        --region "$AWS_REGION")

    FUNCTION_ARN=$(echo "$FUNCTION_INFO" | jq -r '.Configuration.FunctionArn')
    LAST_MODIFIED=$(echo "$FUNCTION_INFO" | jq -r '.Configuration.LastModified')
    CODE_SIZE=$(echo "$FUNCTION_INFO" | jq -r '.Configuration.CodeSize')

    log_success "Deployment completed successfully!"
    echo ""
    echo "Function Details:"
    echo "  Name:          $FUNCTION_NAME"
    echo "  ARN:           $FUNCTION_ARN"
    echo "  Region:        $AWS_REGION"
    echo "  Memory:        ${MEMORY_SIZE}MB"
    echo "  Timeout:       ${TIMEOUT}s"
    echo "  Image:         $FULL_IMAGE_NAME"
    echo "  Code Size:     $(numfmt --to=iec-i --suffix=B $CODE_SIZE)"
    echo "  Last Modified: $LAST_MODIFIED"
    echo ""
    echo "Next Steps:"
    echo "  1. Test function: aws lambda invoke --function-name $FUNCTION_NAME output.json"
    echo "  2. View logs: aws logs tail /aws/lambda/$FUNCTION_NAME --follow"
    echo "  3. Create API Gateway: Use AWS Console or CLI"
else
    log_info "DRY RUN completed - no changes made"
fi

print_header "Deployment Complete"
