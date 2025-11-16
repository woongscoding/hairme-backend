#!/bin/bash
################################################################################
# AWS Infrastructure Cleanup Script for HairMe Backend
#
# This script safely removes unnecessary AWS resources after DynamoDB migration:
# - RDS MySQL instances
# - Application Load Balancers (ALB)
# - NAT Gateways
# - VPC resources (Subnets, Route Tables, Internet Gateways)
#
# ⚠️  DANGER: This script deletes AWS resources. Use with extreme caution!
#
# Usage:
#   ./scripts/cleanup_infrastructure.sh [OPTIONS]
#
# Options:
#   --dry-run              Show what would be deleted without executing
#   --region REGION        AWS region (default: ap-northeast-2)
#   --skip-rds             Skip RDS deletion
#   --skip-alb             Skip ALB deletion
#   --skip-nat             Skip NAT Gateway deletion
#   --skip-vpc             Skip VPC deletion
#   --no-snapshot          Skip RDS snapshot creation (NOT RECOMMENDED)
#   --auto-approve         Skip confirmation prompts (DANGEROUS)
#
# Examples:
#   # Dry run (safe)
#   ./scripts/cleanup_infrastructure.sh --dry-run
#
#   # Delete only RDS
#   ./scripts/cleanup_infrastructure.sh --skip-alb --skip-nat --skip-vpc
#
# Prerequisites:
#   - AWS CLI installed and configured
#   - IAM permissions for RDS, EC2, ELB operations
#   - DynamoDB migration completed and verified
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable

# ==================== Configuration ====================

# Default values
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
DRY_RUN=false
SKIP_RDS=false
SKIP_ALB=false
SKIP_NAT=false
SKIP_VPC=false
NO_SNAPSHOT=false
AUTO_APPROVE=false

# Log file
LOG_FILE="cleanup_log_$(date +%Y%m%d_%H%M%S).json"
DELETION_LOG="deletion_history_$(date +%Y%m%d_%H%M%S).txt"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Tracking
DELETED_RESOURCES=()
FAILED_DELETIONS=()
SKIPPED_RESOURCES=()

# ==================== Parse Arguments ====================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --region)
            AWS_REGION="$2"
            shift 2
            ;;
        --skip-rds)
            SKIP_RDS=true
            shift
            ;;
        --skip-alb)
            SKIP_ALB=true
            shift
            ;;
        --skip-nat)
            SKIP_NAT=true
            shift
            ;;
        --skip-vpc)
            SKIP_VPC=true
            shift
            ;;
        --no-snapshot)
            NO_SNAPSHOT=true
            shift
            ;;
        --auto-approve)
            AUTO_APPROVE=true
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
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$DELETION_LOG"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] SUCCESS: $1" >> "$DELETION_LOG"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$DELETION_LOG"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$DELETION_LOG"
}

log_danger() {
    echo -e "${MAGENTA}🔥 DANGER: $1${NC}"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] DANGER: $1" >> "$DELETION_LOG"
}

print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

confirm_action() {
    local message="$1"
    local countdown="${2:-0}"

    if [ "$AUTO_APPROVE" = true ]; then
        log_warning "Auto-approve enabled - skipping confirmation"
        return 0
    fi

    echo ""
    log_danger "$message"

    if [ "$countdown" -gt 0 ]; then
        log_warning "Waiting ${countdown} seconds... (Press Ctrl+C to cancel)"
        for i in $(seq "$countdown" -1 1); do
            echo -n "$i... "
            sleep 1
        done
        echo ""
    fi

    read -p "Continue? (yes/no): " -r
    echo
    if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
        log_info "Operation cancelled by user"
        return 1
    fi
    return 0
}

# ==================== Validation ====================

print_header "AWS Infrastructure Cleanup"

log_info "Region: $AWS_REGION"

if [ "$DRY_RUN" = true ]; then
    log_warning "🔍 DRY RUN MODE - No resources will be deleted"
fi

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

# ==================== Warning Banner ====================

if [ "$DRY_RUN" = false ]; then
    print_header "⚠️  WARNING ⚠️"
    echo ""
    log_danger "This script will DELETE AWS resources"
    log_danger "Deleted resources CANNOT be easily recovered"
    log_danger "Ensure DynamoDB migration is VERIFIED before proceeding"
    echo ""
    echo "Resources to be deleted:"
    [ "$SKIP_RDS" = false ] && echo "  - RDS MySQL instances"
    [ "$SKIP_ALB" = false ] && echo "  - Application Load Balancers"
    [ "$SKIP_NAT" = false ] && echo "  - NAT Gateways"
    [ "$SKIP_VPC" = false ] && echo "  - VPC resources"
    echo ""

    if ! confirm_action "Do you want to proceed with infrastructure cleanup?" 5; then
        log_info "Cleanup cancelled"
        exit 0
    fi
fi

# ==================== RDS Cleanup ====================

if [ "$SKIP_RDS" = false ]; then
    print_header "RDS MySQL Cleanup"

    # List RDS instances
    log_info "Searching for RDS instances..."

    RDS_INSTANCES=$(aws rds describe-db-instances \
        --region "$AWS_REGION" \
        --query 'DBInstances[*].[DBInstanceIdentifier,DBInstanceStatus,Engine,DBInstanceClass,AllocatedStorage]' \
        --output text 2>/dev/null || echo "")

    if [ -z "$RDS_INSTANCES" ]; then
        log_info "No RDS instances found"
    else
        echo ""
        log_info "Found RDS instances:"
        echo "$RDS_INSTANCES" | while IFS=$'\t' read -r id status engine class storage; do
            echo "  - $id ($status) - $engine $class ${storage}GB"
        done
        echo ""

        # Process each RDS instance
        echo "$RDS_INSTANCES" | while IFS=$'\t' read -r db_id status engine class storage; do

            log_info "Processing RDS instance: $db_id"

            # Create snapshot (unless --no-snapshot)
            if [ "$NO_SNAPSHOT" = false ] && [ "$DRY_RUN" = false ]; then
                SNAPSHOT_ID="${db_id}-final-snapshot-$(date +%Y%m%d-%H%M%S)"

                log_info "Creating final snapshot: $SNAPSHOT_ID"

                aws rds create-db-snapshot \
                    --db-instance-identifier "$db_id" \
                    --db-snapshot-identifier "$SNAPSHOT_ID" \
                    --region "$AWS_REGION" \
                    --output json > /dev/null

                log_success "Snapshot creation initiated: $SNAPSHOT_ID"

                # Wait for snapshot to complete
                log_info "Waiting for snapshot to complete (this may take 5-15 minutes)..."
                aws rds wait db-snapshot-completed \
                    --db-snapshot-identifier "$SNAPSHOT_ID" \
                    --region "$AWS_REGION"

                log_success "Snapshot completed: $SNAPSHOT_ID"

            elif [ "$NO_SNAPSHOT" = true ]; then
                log_warning "Skipping snapshot creation (--no-snapshot flag)"
            fi

            # Delete RDS instance
            if confirm_action "Delete RDS instance '$db_id'?" 3; then

                if [ "$DRY_RUN" = false ]; then
                    log_info "Deleting RDS instance: $db_id"

                    aws rds delete-db-instance \
                        --db-instance-identifier "$db_id" \
                        --skip-final-snapshot \
                        --region "$AWS_REGION" \
                        --output json > /dev/null

                    log_success "RDS deletion initiated: $db_id"
                    DELETED_RESOURCES+=("RDS:$db_id")

                    log_info "Waiting for deletion to complete..."
                    aws rds wait db-instance-deleted \
                        --db-instance-identifier "$db_id" \
                        --region "$AWS_REGION" || true

                    log_success "RDS instance deleted: $db_id"
                else
                    log_info "[DRY RUN] Would delete RDS instance: $db_id"
                fi
            else
                log_info "Skipped RDS deletion: $db_id"
                SKIPPED_RESOURCES+=("RDS:$db_id")
            fi
        done
    fi
else
    log_info "Skipping RDS cleanup (--skip-rds flag)"
fi

# ==================== ALB Cleanup ====================

if [ "$SKIP_ALB" = false ]; then
    print_header "Application Load Balancer Cleanup"

    # List ALBs
    log_info "Searching for Application Load Balancers..."

    ALB_ARNS=$(aws elbv2 describe-load-balancers \
        --region "$AWS_REGION" \
        --query 'LoadBalancers[?Type==`application`].[LoadBalancerArn,LoadBalancerName,DNSName,State.Code]' \
        --output text 2>/dev/null || echo "")

    if [ -z "$ALB_ARNS" ]; then
        log_info "No Application Load Balancers found"
    else
        echo ""
        log_info "Found ALBs:"
        echo "$ALB_ARNS" | while IFS=$'\t' read -r arn name dns state; do
            echo "  - $name ($state)"
            echo "    DNS: $dns"
        done
        echo ""

        # Process each ALB
        echo "$ALB_ARNS" | while IFS=$'\t' read -r alb_arn alb_name dns state; do

            log_info "Processing ALB: $alb_name"

            # List listeners
            LISTENERS=$(aws elbv2 describe-listeners \
                --load-balancer-arn "$alb_arn" \
                --region "$AWS_REGION" \
                --query 'Listeners[*].[ListenerArn,Port,Protocol]' \
                --output text 2>/dev/null || echo "")

            if [ -n "$LISTENERS" ]; then
                echo "  Listeners:"
                echo "$LISTENERS" | while IFS=$'\t' read -r listener_arn port protocol; do
                    echo "    - Port $port ($protocol)"
                done
            fi

            # List target groups
            TARGET_GROUPS=$(aws elbv2 describe-target-groups \
                --load-balancer-arn "$alb_arn" \
                --region "$AWS_REGION" \
                --query 'TargetGroups[*].[TargetGroupArn,TargetGroupName]' \
                --output text 2>/dev/null || echo "")

            # Delete ALB
            if confirm_action "Delete ALB '$alb_name' and associated listeners?" 3; then

                if [ "$DRY_RUN" = false ]; then
                    log_info "Deleting ALB: $alb_name"

                    aws elbv2 delete-load-balancer \
                        --load-balancer-arn "$alb_arn" \
                        --region "$AWS_REGION" \
                        --output json > /dev/null

                    log_success "ALB deleted: $alb_name"
                    DELETED_RESOURCES+=("ALB:$alb_name")

                    # Delete target groups
                    if [ -n "$TARGET_GROUPS" ]; then
                        log_info "Waiting 30 seconds before deleting target groups..."
                        sleep 30

                        echo "$TARGET_GROUPS" | while IFS=$'\t' read -r tg_arn tg_name; do
                            log_info "Deleting target group: $tg_name"

                            aws elbv2 delete-target-group \
                                --target-group-arn "$tg_arn" \
                                --region "$AWS_REGION" \
                                --output json > /dev/null || log_warning "Failed to delete target group: $tg_name"

                            log_success "Target group deleted: $tg_name"
                        done
                    fi
                else
                    log_info "[DRY RUN] Would delete ALB: $alb_name"
                fi
            else
                log_info "Skipped ALB deletion: $alb_name"
                SKIPPED_RESOURCES+=("ALB:$alb_name")
            fi
        done
    fi
else
    log_info "Skipping ALB cleanup (--skip-alb flag)"
fi

# ==================== NAT Gateway Cleanup ====================

if [ "$SKIP_NAT" = false ]; then
    print_header "NAT Gateway Cleanup"

    # List NAT Gateways
    log_info "Searching for NAT Gateways..."

    NAT_GATEWAYS=$(aws ec2 describe-nat-gateways \
        --region "$AWS_REGION" \
        --filter "Name=state,Values=available,pending" \
        --query 'NatGateways[*].[NatGatewayId,State,SubnetId,VpcId]' \
        --output text 2>/dev/null || echo "")

    if [ -z "$NAT_GATEWAYS" ]; then
        log_info "No NAT Gateways found"
    else
        echo ""
        log_info "Found NAT Gateways:"
        echo "$NAT_GATEWAYS" | while IFS=$'\t' read -r nat_id state subnet vpc; do
            echo "  - $nat_id ($state) in VPC $vpc"
        done
        echo ""

        # Calculate cost
        NAT_COUNT=$(echo "$NAT_GATEWAYS" | wc -l)
        MONTHLY_COST=$(echo "$NAT_COUNT * 33" | bc)
        log_warning "Current NAT Gateway cost: ~\$${MONTHLY_COST}/month"

        # Process each NAT Gateway
        echo "$NAT_GATEWAYS" | while IFS=$'\t' read -r nat_id state subnet vpc; do

            log_info "Processing NAT Gateway: $nat_id"

            # Get associated Elastic IP
            EIP_ALLOCATION=$(aws ec2 describe-nat-gateways \
                --nat-gateway-ids "$nat_id" \
                --region "$AWS_REGION" \
                --query 'NatGateways[0].NatGatewayAddresses[0].AllocationId' \
                --output text 2>/dev/null || echo "")

            if [ -n "$EIP_ALLOCATION" ] && [ "$EIP_ALLOCATION" != "None" ]; then
                log_info "Associated Elastic IP: $EIP_ALLOCATION"
            fi

            # Delete NAT Gateway
            if confirm_action "Delete NAT Gateway '$nat_id'? (Saves ~\$33/month)" 3; then

                if [ "$DRY_RUN" = false ]; then
                    log_info "Deleting NAT Gateway: $nat_id"

                    aws ec2 delete-nat-gateway \
                        --nat-gateway-id "$nat_id" \
                        --region "$AWS_REGION" \
                        --output json > /dev/null

                    log_success "NAT Gateway deletion initiated: $nat_id"
                    DELETED_RESOURCES+=("NAT:$nat_id")

                    log_info "Waiting for NAT Gateway to be deleted (this may take 3-5 minutes)..."

                    # Wait for deletion
                    while true; do
                        NAT_STATE=$(aws ec2 describe-nat-gateways \
                            --nat-gateway-ids "$nat_id" \
                            --region "$AWS_REGION" \
                            --query 'NatGateways[0].State' \
                            --output text 2>/dev/null || echo "deleted")

                        if [ "$NAT_STATE" = "deleted" ]; then
                            break
                        fi

                        echo -n "."
                        sleep 10
                    done
                    echo ""

                    log_success "NAT Gateway deleted: $nat_id"

                    # Release Elastic IP
                    if [ -n "$EIP_ALLOCATION" ] && [ "$EIP_ALLOCATION" != "None" ]; then
                        log_info "Releasing Elastic IP: $EIP_ALLOCATION"

                        aws ec2 release-address \
                            --allocation-id "$EIP_ALLOCATION" \
                            --region "$AWS_REGION" \
                            --output json > /dev/null || log_warning "Failed to release Elastic IP"

                        log_success "Elastic IP released: $EIP_ALLOCATION"
                    fi
                else
                    log_info "[DRY RUN] Would delete NAT Gateway: $nat_id"
                fi
            else
                log_info "Skipped NAT Gateway deletion: $nat_id"
                SKIPPED_RESOURCES+=("NAT:$nat_id")
            fi
        done
    fi
else
    log_info "Skipping NAT Gateway cleanup (--skip-nat flag)"
fi

# ==================== VPC Cleanup ====================

if [ "$SKIP_VPC" = false ]; then
    print_header "VPC Resources Cleanup"

    log_warning "VPC cleanup is complex and may fail if resources are in use"
    log_info "Searching for VPCs with 'hairme' tag..."

    # List VPCs with hairme tag
    VPCS=$(aws ec2 describe-vpcs \
        --region "$AWS_REGION" \
        --filters "Name=tag:Name,Values=*hairme*" \
        --query 'Vpcs[*].[VpcId,CidrBlock,State]' \
        --output text 2>/dev/null || echo "")

    if [ -z "$VPCS" ]; then
        log_info "No VPCs with 'hairme' tag found"
    else
        echo ""
        log_info "Found VPCs:"
        echo "$VPCS" | while IFS=$'\t' read -r vpc_id cidr state; do
            echo "  - $vpc_id ($cidr) - $state"
        done
        echo ""

        log_warning "VPC deletion requires manual cleanup of:"
        log_warning "  - EC2 instances must be terminated"
        log_warning "  - Elastic Network Interfaces must be detached"
        log_warning "  - Security Groups must be deleted"
        log_warning "  - Subnets must be deleted"
        log_warning "  - Route Tables must be deleted"
        log_warning "  - Internet Gateways must be detached"
        echo ""
        log_info "For safety, VPC deletion is not automated in this script"
        log_info "To delete VPC manually:"
        echo ""

        echo "$VPCS" | while IFS=$'\t' read -r vpc_id cidr state; do
            echo "  aws ec2 delete-vpc --vpc-id $vpc_id --region $AWS_REGION"
        done

        SKIPPED_RESOURCES+=("VPC:manual-cleanup-required")
    fi
else
    log_info "Skipping VPC cleanup (--skip-vpc flag)"
fi

# ==================== Summary ====================

print_header "Cleanup Summary"

# Save summary to JSON
cat > "$LOG_FILE" <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "region": "$AWS_REGION",
  "dry_run": $DRY_RUN,
  "account_id": "$ACCOUNT_ID",
  "deleted_resources": $(printf '%s\n' "${DELETED_RESOURCES[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]"),
  "failed_deletions": $(printf '%s\n' "${FAILED_DELETIONS[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]"),
  "skipped_resources": $(printf '%s\n' "${SKIPPED_RESOURCES[@]}" | jq -R . | jq -s . 2>/dev/null || echo "[]")
}
EOF

echo ""
if [ "$DRY_RUN" = false ]; then
    log_success "Infrastructure cleanup completed!"
    echo ""
    echo "Deleted Resources:"
    if [ ${#DELETED_RESOURCES[@]} -eq 0 ]; then
        echo "  (none)"
    else
        printf '  %s\n' "${DELETED_RESOURCES[@]}"
    fi

    echo ""
    echo "Skipped Resources:"
    if [ ${#SKIPPED_RESOURCES[@]} -eq 0 ]; then
        echo "  (none)"
    else
        printf '  %s\n' "${SKIPPED_RESOURCES[@]}"
    fi

    if [ ${#FAILED_DELETIONS[@]} -gt 0 ]; then
        echo ""
        log_error "Failed Deletions:"
        printf '  %s\n' "${FAILED_DELETIONS[@]}"
    fi
else
    log_info "DRY RUN completed - no resources were deleted"
fi

echo ""
echo "Logs saved:"
echo "  Summary:  $LOG_FILE"
echo "  Details:  $DELETION_LOG"

echo ""
echo "Next Steps:"
echo "  1. Verify cleanup: ./scripts/verify_cleanup.sh"
echo "  2. Check AWS billing dashboard for cost reduction"
echo "  3. Monitor for any unexpected issues"
echo "  4. If problems occur, see ROLLBACK.md for recovery"

print_header "Cleanup Complete"
