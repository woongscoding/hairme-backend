#!/bin/bash
################################################################################
# AWS Infrastructure Cleanup Verification Script
#
# This script verifies infrastructure cleanup and estimates remaining costs.
#
# Usage:
#   ./scripts/verify_cleanup.sh [OPTIONS]
#
# Options:
#   --region REGION        AWS region (default: ap-northeast-2)
#   --detailed             Show detailed resource information
#   --export-csv           Export cost report to CSV
#
# Examples:
#   # Basic verification
#   ./scripts/verify_cleanup.sh
#
#   # Detailed verification with CSV export
#   ./scripts/verify_cleanup.sh --detailed --export-csv
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==================== Configuration ====================

# Default values
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
DETAILED=false
EXPORT_CSV=false

# Report files
REPORT_FILE="cleanup_verification_$(date +%Y%m%d_%H%M%S).json"
CSV_FILE="cost_report_$(date +%Y%m%d_%H%M%S).csv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Cost tracking
TOTAL_MONTHLY_COST=0

# ==================== Parse Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --detailed)
            DETAILED=true
            shift
            ;;
        --export-csv)
            EXPORT_CSV=true
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

log_cost() {
    echo -e "${MAGENTA}💰 $1${NC}"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

add_cost() {
    local cost=$1
    TOTAL_MONTHLY_COST=$(echo "$TOTAL_MONTHLY_COST + $cost" | bc)
}

format_cost() {
    local cost=$1
    printf "\$%.2f" "$cost"
}

# ==================== Validation ====================

print_header "Infrastructure Cleanup Verification"

log_info "Region: $AWS_REGION"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not installed"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    log_error "AWS credentials not configured"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
log_info "AWS Account: $ACCOUNT_ID"

log_success "Prerequisites check passed"

# Initialize CSV if needed
if [ "$EXPORT_CSV" = true ]; then
    echo "Resource Type,Resource ID,Status,Monthly Cost (USD),Annual Cost (USD)" > "$CSV_FILE"
    log_info "CSV export enabled: $CSV_FILE"
fi

# ==================== RDS Verification ====================

print_header "RDS MySQL Instances"

RDS_INSTANCES=$(aws rds describe-db-instances \
    --region "$AWS_REGION" \
    --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceStatus,Engine,DBInstanceClass,AllocatedStorage,MultiAZ]' \
    --output text 2>/dev/null || echo "")

if [ -z "$RDS_INSTANCES" ]; then
    log_success "✓ No RDS instances found (expected after cleanup)"
    RDS_COUNT=0
    RDS_COST=0
else
    RDS_COUNT=$(echo "$RDS_INSTANCES" | wc -l)
    log_warning "Found $RDS_COUNT RDS instance(s) - cleanup may be incomplete"

    echo ""
    echo "$RDS_INSTANCES" | while IFS=$'\t' read -r id status engine class storage multi_az; do
        echo "  - $id ($status)"
        echo "    Engine: $engine"
        echo "    Instance: $class"
        echo "    Storage: ${storage}GB"
        echo "    Multi-AZ: $multi_az"

        # Estimate cost (rough estimate based on db.t3.micro)
        case $class in
            db.t3.micro)
                INSTANCE_COST=12
                ;;
            db.t3.small)
                INSTANCE_COST=24
                ;;
            db.t3.medium)
                INSTANCE_COST=48
                ;;
            db.t4g.micro)
                INSTANCE_COST=10
                ;;
            *)
                INSTANCE_COST=20
                ;;
        esac

        STORAGE_COST=$(echo "$storage * 0.115" | bc)
        TOTAL_RDS_COST=$(echo "$INSTANCE_COST + $STORAGE_COST" | bc)

        if [ "$multi_az" = "True" ]; then
            TOTAL_RDS_COST=$(echo "$TOTAL_RDS_COST * 2" | bc)
        fi

        log_cost "Estimated cost: $(format_cost $TOTAL_RDS_COST)/month"
        add_cost "$TOTAL_RDS_COST"

        if [ "$EXPORT_CSV" = true ]; then
            ANNUAL=$(echo "$TOTAL_RDS_COST * 12" | bc)
            echo "RDS,$id,active,$(format_cost $TOTAL_RDS_COST),$(format_cost $ANNUAL)" >> "$CSV_FILE"
        fi

        echo ""
    done

    RDS_COST=$TOTAL_RDS_COST
fi

# Check RDS snapshots
RDS_SNAPSHOTS=$(aws rds describe-db-snapshots \
    --region "$AWS_REGION" \
    --query 'DBSnapshots[*].[DBSnapshotIdentifier,Status,AllocatedStorage,SnapshotCreateTime]' \
    --output text 2>/dev/null || echo "")

if [ -n "$RDS_SNAPSHOTS" ]; then
    SNAPSHOT_COUNT=$(echo "$RDS_SNAPSHOTS" | wc -l)
    log_info "Found $SNAPSHOT_COUNT RDS snapshot(s)"

    if [ "$DETAILED" = true ]; then
        echo ""
        echo "$RDS_SNAPSHOTS" | while IFS=$'\t' read -r id status storage created; do
            echo "  - $id ($status) - ${storage}GB"
            echo "    Created: $created"
        done
    fi

    # Calculate snapshot storage cost
    TOTAL_SNAPSHOT_STORAGE=$(echo "$RDS_SNAPSHOTS" | awk '{sum+=$3} END {print sum}')
    SNAPSHOT_COST=$(echo "$TOTAL_SNAPSHOT_STORAGE * 0.095" | bc)

    log_cost "Snapshot storage cost: $(format_cost $SNAPSHOT_COST)/month"
    add_cost "$SNAPSHOT_COST"

    if [ "$EXPORT_CSV" = true ]; then
        ANNUAL=$(echo "$SNAPSHOT_COST * 12" | bc)
        echo "RDS Snapshots,${SNAPSHOT_COUNT} snapshots,active,$(format_cost $SNAPSHOT_COST),$(format_cost $ANNUAL)" >> "$CSV_FILE"
    fi
else
    log_info "No RDS snapshots found"
fi

# ==================== ALB Verification ====================

print_header "Application Load Balancers"

ALB_ARNS=$(aws elbv2 describe-load-balancers \
    --region "$AWS_REGION" \
    --query 'LoadBalancers[?Type==`application`].[LoadBalancerArn,LoadBalancerName,DNSName,State.Code]' \
    --output text 2>/dev/null || echo "")

if [ -z "$ALB_ARNS" ]; then
    log_success "✓ No ALBs found (expected after cleanup)"
    ALB_COUNT=0
else
    ALB_COUNT=$(echo "$ALB_ARNS" | wc -l)
    log_warning "Found $ALB_COUNT ALB(s) - cleanup may be incomplete"

    echo ""
    echo "$ALB_ARNS" | while IFS=$'\t' read -r arn name dns state; do
        echo "  - $name ($state)"
        echo "    DNS: $dns"

        # ALB cost: ~$16.20/month (720 hours)
        ALB_COST=16.20

        log_cost "Estimated cost: $(format_cost $ALB_COST)/month"
        add_cost "$ALB_COST"

        if [ "$EXPORT_CSV" = true ]; then
            ANNUAL=$(echo "$ALB_COST * 12" | bc)
            echo "ALB,$name,active,$(format_cost $ALB_COST),$(format_cost $ANNUAL)" >> "$CSV_FILE"
        fi

        echo ""
    done
fi

# ==================== NAT Gateway Verification ====================

print_header "NAT Gateways"

NAT_GATEWAYS=$(aws ec2 describe-nat-gateways \
    --region "$AWS_REGION" \
    --filter "Name=state,Values=available,pending" \
    --query 'NatGateways[*].[NatGatewayId,State,SubnetId,VpcId]' \
    --output text 2>/dev/null || echo "")

if [ -z "$NAT_GATEWAYS" ]; then
    log_success "✓ No NAT Gateways found (expected after cleanup)"
    NAT_COUNT=0
else
    NAT_COUNT=$(echo "$NAT_GATEWAYS" | wc -l)
    log_warning "Found $NAT_COUNT NAT Gateway(s) - cleanup may be incomplete"

    echo ""
    echo "$NAT_GATEWAYS" | while IFS=$'\t' read -r nat_id state subnet vpc; do
        echo "  - $nat_id ($state)"
        echo "    VPC: $vpc"
        echo "    Subnet: $subnet"

        # NAT Gateway cost: ~$32.85/month
        NAT_COST=32.85

        log_cost "Estimated cost: $(format_cost $NAT_COST)/month"
        add_cost "$NAT_COST"

        if [ "$EXPORT_CSV" = true ]; then
            ANNUAL=$(echo "$NAT_COST * 12" | bc)
            echo "NAT Gateway,$nat_id,active,$(format_cost $NAT_COST),$(format_cost $ANNUAL)" >> "$CSV_FILE"
        fi

        echo ""
    done
fi

# ==================== Elastic IP Verification ====================

print_header "Elastic IPs"

# Unassociated Elastic IPs cost money
UNASSOCIATED_EIPS=$(aws ec2 describe-addresses \
    --region "$AWS_REGION" \
    --query 'Addresses[?AssociationId==`null`].[AllocationId,PublicIp]' \
    --output text 2>/dev/null || echo "")

if [ -z "$UNASSOCIATED_EIPS" ]; then
    log_success "✓ No unassociated Elastic IPs found"
else
    EIP_COUNT=$(echo "$UNASSOCIATED_EIPS" | wc -l)
    log_warning "Found $EIP_COUNT unassociated Elastic IP(s)"

    echo ""
    echo "$UNASSOCIATED_EIPS" | while IFS=$'\t' read -r alloc_id public_ip; do
        echo "  - $public_ip (Allocation: $alloc_id)"

        # Unassociated EIP cost: ~$3.60/month
        EIP_COST=3.60

        log_cost "Estimated cost: $(format_cost $EIP_COST)/month"
        add_cost "$EIP_COST"

        if [ "$EXPORT_CSV" = true ]; then
            ANNUAL=$(echo "$EIP_COST * 12" | bc)
            echo "Elastic IP,$public_ip,unassociated,$(format_cost $EIP_COST),$(format_cost $ANNUAL)" >> "$CSV_FILE"
        fi
    done
    echo ""

    log_info "Release unused EIPs with:"
    echo "$UNASSOCIATED_EIPS" | while IFS=$'\t' read -r alloc_id public_ip; do
        echo "  aws ec2 release-address --allocation-id $alloc_id --region $AWS_REGION"
    done
fi

# ==================== Lambda Verification ====================

print_header "Lambda Functions"

LAMBDA_FUNCTIONS=$(aws lambda list-functions \
    --region "$AWS_REGION" \
    --query 'Functions[?contains(FunctionName, `hairme`)].[FunctionName,Runtime,MemorySize,CodeSize]' \
    --output text 2>/dev/null || echo "")

if [ -z "$LAMBDA_FUNCTIONS" ]; then
    log_info "No Lambda functions found"
    LAMBDA_COUNT=0
else
    LAMBDA_COUNT=$(echo "$LAMBDA_FUNCTIONS" | wc -l)
    log_success "Found $LAMBDA_COUNT Lambda function(s) (new architecture)"

    if [ "$DETAILED" = true ]; then
        echo ""
        echo "$LAMBDA_FUNCTIONS" | while IFS=$'\t' read -r name runtime memory code_size; do
            echo "  - $name"
            echo "    Runtime: $runtime"
            echo "    Memory: ${memory}MB"
            echo "    Code Size: $(numfmt --to=iec-i --suffix=B $code_size 2>/dev/null || echo ${code_size}B)"
        done
        echo ""
    fi

    log_info "Lambda cost is typically very low (free tier: 1M requests/month)"
    log_cost "Estimated cost: \$0.00 - \$5.00/month (depends on usage)"
fi

# ==================== DynamoDB Verification ====================

print_header "DynamoDB Tables"

DYNAMODB_TABLES=$(aws dynamodb list-tables \
    --region "$AWS_REGION" \
    --query 'TableNames' \
    --output text 2>/dev/null || echo "")

if [ -z "$DYNAMODB_TABLES" ]; then
    log_warning "No DynamoDB tables found - migration may not be complete"
else
    DYNAMODB_COUNT=$(echo "$DYNAMODB_TABLES" | wc -w)
    log_success "Found $DYNAMODB_COUNT DynamoDB table(s) (new architecture)"

    if [ "$DETAILED" = true ]; then
        echo ""
        for table in $DYNAMODB_TABLES; do
            TABLE_INFO=$(aws dynamodb describe-table \
                --table-name "$table" \
                --region "$AWS_REGION" \
                --query 'Table.[TableName,TableStatus,ItemCount,TableSizeBytes,BillingModeSummary.BillingMode]' \
                --output text 2>/dev/null || echo "")

            if [ -n "$TABLE_INFO" ]; then
                echo "$TABLE_INFO" | while IFS=$'\t' read -r name status items size billing; do
                    echo "  - $name ($status)"
                    echo "    Items: $items"
                    echo "    Size: $(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo ${size}B)"
                    echo "    Billing: $billing"
                done
            fi
        done
        echo ""
    fi

    log_info "DynamoDB on-demand pricing: pay per request"
    log_cost "Estimated cost: \$0.00 - \$2.00/month (within free tier for low usage)"
fi

# ==================== VPC Verification ====================

print_header "VPC Resources"

# Count VPCs
VPCS=$(aws ec2 describe-vpcs \
    --region "$AWS_REGION" \
    --filters "Name=tag:Name,Values=*hairme*" \
    --query 'Vpcs[*].[VpcId,CidrBlock,State]' \
    --output text 2>/dev/null || echo "")

if [ -z "$VPCS" ]; then
    log_info "No VPCs with 'hairme' tag found"
else
    VPC_COUNT=$(echo "$VPCS" | wc -l)
    log_info "Found $VPC_COUNT VPC(s) with 'hairme' tag"

    if [ "$DETAILED" = true ]; then
        echo ""
        echo "$VPCS" | while IFS=$'\t' read -r vpc_id cidr state; do
            echo "  - $vpc_id ($cidr) - $state"
        done
        echo ""
    fi

    log_info "VPC resources are free (no charge for VPCs, subnets, route tables)"
fi

# ==================== Cost Comparison ====================

print_header "Cost Analysis"

echo ""
log_info "Previous Monthly Cost (RDS + ALB + NAT):"
echo "  RDS MySQL (db.t3.micro):     ~\$12.00"
echo "  ALB:                         ~\$16.20"
echo "  NAT Gateway (if used):       ~\$32.85"
echo "  ───────────────────────────────────"
echo "  Total (Old):                 ~\$61.05/month"

echo ""
log_info "Current Monthly Cost (DynamoDB + Lambda):"
if [ "$TOTAL_MONTHLY_COST" = "0" ] || [ -z "$TOTAL_MONTHLY_COST" ]; then
    echo "  DynamoDB (on-demand):        ~\$0.00 (free tier)"
    echo "  Lambda:                      ~\$0.00 (free tier)"
    echo "  ───────────────────────────────────"
    log_success "Total (New):                 ~\$0.00/month"
    echo ""
    log_success "💰 Monthly Savings: ~\$61.05 (~\$733/year)"
else
    echo "  Active Resources:            ~$(format_cost $TOTAL_MONTHLY_COST)"
    echo "  DynamoDB (on-demand):        ~\$0.00 (free tier)"
    echo "  Lambda:                      ~\$0.00 (free tier)"
    echo "  ───────────────────────────────────"
    log_warning "Total (New):                 ~$(format_cost $TOTAL_MONTHLY_COST)/month"
    echo ""

    SAVINGS=$(echo "61.05 - $TOTAL_MONTHLY_COST" | bc)
    ANNUAL_SAVINGS=$(echo "$SAVINGS * 12" | bc)

    if (( $(echo "$SAVINGS > 0" | bc -l) )); then
        log_cost "💰 Monthly Savings: ~$(format_cost $SAVINGS) (~$(format_cost $ANNUAL_SAVINGS)/year)"
    else
        log_warning "⚠️  Cost increased by $(format_cost ${SAVINGS#-})/month"
    fi
fi

# ==================== Recommendations ====================

print_header "Recommendations"

echo ""

# Check for remaining old infrastructure
if [ "$RDS_COUNT" -gt 0 ]; then
    log_warning "RDS instances still exist"
    echo "  → Run cleanup script: ./scripts/cleanup_infrastructure.sh"
fi

if [ "$ALB_COUNT" -gt 0 ]; then
    log_warning "ALBs still exist"
    echo "  → Run cleanup script: ./scripts/cleanup_infrastructure.sh --skip-rds"
fi

if [ "$NAT_COUNT" -gt 0 ]; then
    log_warning "NAT Gateways still exist (costly!)"
    echo "  → Run cleanup script: ./scripts/cleanup_infrastructure.sh --skip-rds --skip-alb"
fi

# Check for new infrastructure
if [ "$LAMBDA_COUNT" -eq 0 ]; then
    log_warning "No Lambda functions found"
    echo "  → Deploy Lambda: ./scripts/deploy_lambda.sh"
fi

if [ -z "$DYNAMODB_TABLES" ]; then
    log_error "No DynamoDB tables found - migration not complete!"
    echo "  → Create table: ./scripts/create_dynamodb_table.sh"
    echo "  → Run migration: python scripts/migrate_rds_to_dynamodb.py"
fi

# Final status
echo ""
if [ "$TOTAL_MONTHLY_COST" = "0" ] || [ -z "$TOTAL_MONTHLY_COST" ]; then
    log_success "✓ Infrastructure cleanup successful!"
    log_success "✓ All expensive resources removed"
    log_success "✓ Running on serverless architecture (DynamoDB + Lambda)"
else
    log_warning "⚠️  Cleanup incomplete - some resources still active"
    echo "  Current monthly cost: ~$(format_cost $TOTAL_MONTHLY_COST)"
fi

# ==================== Save Report ====================

cat > "$REPORT_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "region": "$AWS_REGION",
  "account_id": "$ACCOUNT_ID",
  "summary": {
    "rds_instances": $RDS_COUNT,
    "alb_count": $ALB_COUNT,
    "nat_gateways": $NAT_COUNT,
    "lambda_functions": $LAMBDA_COUNT,
    "dynamodb_tables": $DYNAMODB_COUNT
  },
  "costs": {
    "monthly_total_usd": $TOTAL_MONTHLY_COST,
    "annual_total_usd": $(echo "$TOTAL_MONTHLY_COST * 12" | bc),
    "old_monthly_usd": 61.05,
    "monthly_savings_usd": $(echo "61.05 - $TOTAL_MONTHLY_COST" | bc),
    "annual_savings_usd": $(echo "(61.05 - $TOTAL_MONTHLY_COST) * 12" | bc)
  },
  "cleanup_status": "$([ "$TOTAL_MONTHLY_COST" = "0" ] && echo "complete" || echo "incomplete")"
}
EOF

log_success "Verification report saved: $REPORT_FILE"

if [ "$EXPORT_CSV" = true ]; then
    log_success "Cost report CSV saved: $CSV_FILE"
fi

print_header "Verification Complete"
