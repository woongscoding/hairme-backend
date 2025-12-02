#!/usr/bin/env python3
"""
MLOps Lambda 배포 스크립트

1. hairme-model-trainer Lambda 업데이트
2. hairme-ab-evaluator Lambda 생성/업데이트
3. EventBridge 규칙 생성

Usage:
    python scripts/deploy_mlops_lambdas.py [--trainer-only] [--evaluator-only] [--rules-only]
"""

import argparse
import boto3
import json
import zipfile
import io
import os
from datetime import datetime

# Configuration
AWS_REGION = 'ap-northeast-2'
ACCOUNT_ID = '364042451408'

# Lambda Configuration
TRAINER_LAMBDA_NAME = 'hairme-model-trainer'
EVALUATOR_LAMBDA_NAME = 'hairme-ab-evaluator'

TRAINER_ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/hairme-trainer-role'
EVALUATOR_ROLE_ARN = f'arn:aws:iam::{ACCOUNT_ID}:role/hairme-evaluator-role'

# EventBridge Configuration
EVALUATOR_RULE_NAME = 'hairme-daily-evaluation'
EVALUATOR_SCHEDULE = 'cron(0 15 * * ? *)'  # 매일 UTC 15:00 (KST 00:00)


def get_lambda_client():
    return boto3.client('lambda', region_name=AWS_REGION)


def get_events_client():
    return boto3.client('events', region_name=AWS_REGION)


def get_iam_client():
    return boto3.client('iam', region_name=AWS_REGION)


def create_zip_from_file(source_file: str) -> bytes:
    """단일 파일에서 ZIP 생성"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, 'lambda_function.py')
    buffer.seek(0)
    return buffer.read()


def ensure_iam_role(role_name: str, role_arn: str) -> bool:
    """IAM Role 존재 확인 및 생성"""
    iam = get_iam_client()

    try:
        iam.get_role(RoleName=role_name)
        print(f"✅ IAM Role exists: {role_name}")
        return True
    except iam.exceptions.NoSuchEntityException:
        print(f"⚠️ IAM Role not found: {role_name}")
        print("   Please create the role manually with the following permissions:")
        print("   - AWSLambdaBasicExecutionRole")
        print("   - S3 read/write to hairme-mlops bucket")
        print("   - DynamoDB read for hairme-analysis table")
        print("   - Lambda invoke for hairme-analyze")
        return False


def deploy_trainer_lambda(dry_run: bool = False):
    """Trainer Lambda 배포"""
    print("\n=== Deploying Trainer Lambda ===")

    source_file = 'lambda_trainer/lambda_function.py'
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would deploy {source_file} to {TRAINER_LAMBDA_NAME}")
        return True

    lambda_client = get_lambda_client()

    try:
        # ZIP 생성
        zip_content = create_zip_from_file(source_file)
        print(f"✅ Created ZIP package ({len(zip_content)} bytes)")

        # Lambda 업데이트
        response = lambda_client.update_function_code(
            FunctionName=TRAINER_LAMBDA_NAME,
            ZipFile=zip_content
        )
        print(f"✅ Lambda code updated: {TRAINER_LAMBDA_NAME}")

        # 환경변수 업데이트
        lambda_client.update_function_configuration(
            FunctionName=TRAINER_LAMBDA_NAME,
            Environment={
                'Variables': {
                    'MLOPS_S3_BUCKET': 'hairme-mlops',
                    'MLOPS_MIN_SAMPLES': '50',
                    'ANALYZE_LAMBDA_NAME': 'hairme-analyze',
                    'AWS_REGION': AWS_REGION,
                    'FINE_TUNE_EPOCHS': '10',
                    'FINE_TUNE_LR': '0.0001',
                    'BATCH_SIZE': '32'
                }
            },
            Timeout=900,  # 15분
            MemorySize=1024
        )
        print(f"✅ Lambda configuration updated")

        return True

    except lambda_client.exceptions.ResourceNotFoundException:
        print(f"⚠️ Lambda function not found: {TRAINER_LAMBDA_NAME}")
        print("   Please create the function first in AWS Console or with:")
        print(f"   aws lambda create-function --function-name {TRAINER_LAMBDA_NAME} ...")
        return False
    except Exception as e:
        print(f"❌ Failed to deploy trainer: {e}")
        return False


def deploy_evaluator_lambda(dry_run: bool = False):
    """Evaluator Lambda 배포"""
    print("\n=== Deploying Evaluator Lambda ===")

    source_file = 'lambda_evaluator/lambda_function.py'
    if not os.path.exists(source_file):
        print(f"❌ Source file not found: {source_file}")
        return False

    if dry_run:
        print(f"[DRY RUN] Would deploy {source_file} to {EVALUATOR_LAMBDA_NAME}")
        return True

    lambda_client = get_lambda_client()

    try:
        # ZIP 생성
        zip_content = create_zip_from_file(source_file)
        print(f"✅ Created ZIP package ({len(zip_content)} bytes)")

        # Lambda 존재 확인
        try:
            lambda_client.get_function(FunctionName=EVALUATOR_LAMBDA_NAME)
            function_exists = True
        except lambda_client.exceptions.ResourceNotFoundException:
            function_exists = False

        env_vars = {
            'MLOPS_S3_BUCKET': 'hairme-mlops',
            'ANALYZE_LAMBDA_NAME': 'hairme-analyze',
            'AWS_REGION': AWS_REGION,
            'DYNAMODB_TABLE_NAME': 'hairme-analysis',
            'EVAL_MIN_SAMPLES': '100',
            'EVAL_MIN_IMPROVEMENT': '0.02',
            'AUTO_PROMOTE': 'true'
        }

        if function_exists:
            # 업데이트
            lambda_client.update_function_code(
                FunctionName=EVALUATOR_LAMBDA_NAME,
                ZipFile=zip_content
            )
            print(f"✅ Lambda code updated: {EVALUATOR_LAMBDA_NAME}")

            lambda_client.update_function_configuration(
                FunctionName=EVALUATOR_LAMBDA_NAME,
                Environment={'Variables': env_vars},
                Timeout=300,  # 5분
                MemorySize=256
            )
            print(f"✅ Lambda configuration updated")
        else:
            # 생성
            print(f"Creating new Lambda function: {EVALUATOR_LAMBDA_NAME}")

            # Role 확인
            if not ensure_iam_role('hairme-evaluator-role', EVALUATOR_ROLE_ARN):
                print("⚠️ Skipping Lambda creation - role not found")
                return False

            lambda_client.create_function(
                FunctionName=EVALUATOR_LAMBDA_NAME,
                Runtime='python3.12',
                Role=EVALUATOR_ROLE_ARN,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Environment={'Variables': env_vars},
                Timeout=300,
                MemorySize=256,
                Description='HairMe A/B Test Evaluator - Daily evaluation and auto-promotion'
            )
            print(f"✅ Lambda created: {EVALUATOR_LAMBDA_NAME}")

        return True

    except Exception as e:
        print(f"❌ Failed to deploy evaluator: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_eventbridge_rule(dry_run: bool = False):
    """Evaluator용 EventBridge 규칙 생성"""
    print("\n=== Creating EventBridge Rule ===")

    if dry_run:
        print(f"[DRY RUN] Would create rule: {EVALUATOR_RULE_NAME}")
        print(f"[DRY RUN] Schedule: {EVALUATOR_SCHEDULE}")
        return True

    events_client = get_events_client()
    lambda_client = get_lambda_client()

    try:
        # 규칙 생성/업데이트
        events_client.put_rule(
            Name=EVALUATOR_RULE_NAME,
            ScheduleExpression=EVALUATOR_SCHEDULE,
            State='ENABLED',
            Description='Daily A/B test evaluation at UTC 15:00 (KST 00:00)'
        )
        print(f"✅ EventBridge rule created: {EVALUATOR_RULE_NAME}")

        # Lambda ARN 가져오기
        lambda_response = lambda_client.get_function(FunctionName=EVALUATOR_LAMBDA_NAME)
        lambda_arn = lambda_response['Configuration']['FunctionArn']

        # Target 설정
        events_client.put_targets(
            Rule=EVALUATOR_RULE_NAME,
            Targets=[
                {
                    'Id': 'evaluator-target',
                    'Arn': lambda_arn,
                    'Input': json.dumps({
                        'trigger_type': 'scheduled',
                        'schedule': 'daily'
                    })
                }
            ]
        )
        print(f"✅ Target added: {EVALUATOR_LAMBDA_NAME}")

        # Lambda permission 추가
        try:
            lambda_client.add_permission(
                FunctionName=EVALUATOR_LAMBDA_NAME,
                StatementId='eventbridge-daily-evaluation',
                Action='lambda:InvokeFunction',
                Principal='events.amazonaws.com',
                SourceArn=f'arn:aws:events:{AWS_REGION}:{ACCOUNT_ID}:rule/{EVALUATOR_RULE_NAME}'
            )
            print(f"✅ Lambda permission added")
        except lambda_client.exceptions.ResourceConflictException:
            print(f"ℹ️ Lambda permission already exists")

        return True

    except Exception as e:
        print(f"❌ Failed to create EventBridge rule: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Deploy MLOps Lambdas')
    parser.add_argument('--trainer-only', action='store_true', help='Deploy trainer only')
    parser.add_argument('--evaluator-only', action='store_true', help='Deploy evaluator only')
    parser.add_argument('--rules-only', action='store_true', help='Create EventBridge rules only')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')

    args = parser.parse_args()

    print("=" * 60)
    print("HairMe MLOps Lambda Deployment")
    print("=" * 60)
    print(f"Region: {AWS_REGION}")
    print(f"Account: {ACCOUNT_ID}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    deploy_all = not (args.trainer_only or args.evaluator_only or args.rules_only)

    success = True

    if deploy_all or args.trainer_only:
        if not deploy_trainer_lambda(args.dry_run):
            success = False

    if deploy_all or args.evaluator_only:
        if not deploy_evaluator_lambda(args.dry_run):
            success = False

    if deploy_all or args.rules_only:
        if not create_eventbridge_rule(args.dry_run):
            success = False

    print("\n" + "=" * 60)
    if success:
        print("✅ Deployment completed successfully!")
    else:
        print("⚠️ Deployment completed with warnings/errors")
    print("=" * 60)

    # 배포 후 안내
    print("\n📋 Next Steps:")
    print("1. Verify Lambda functions in AWS Console")
    print("2. Check CloudWatch logs for any errors")
    print("3. Test manually with:")
    print(f"   aws lambda invoke --function-name {TRAINER_LAMBDA_NAME} \\")
    print('     --payload \'{"trigger_type": "manual", "force": true}\' response.json')
    print()
    print(f"   aws lambda invoke --function-name {EVALUATOR_LAMBDA_NAME} \\")
    print('     --payload \'{"trigger_type": "manual"}\' response.json')


if __name__ == '__main__':
    main()
