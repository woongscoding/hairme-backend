#!/usr/bin/env python3
"""
MLOps 인프라 배포 스크립트

AWS 리소스를 생성하고 MLOps 파이프라인을 설정합니다.

생성되는 리소스:
1. S3 버킷 (hairme-mlops)
2. IAM 역할 (hairme-trainer-role)
3. Lambda 함수 (hairme-model-trainer)
4. EventBridge 규칙 (hairme-weekly-retrain)

비용:
    - S3: $0.023/GB/월 (거의 무료)
    - Lambda: 프리티어 내 (월 100만 요청)
    - EventBridge: 무료

사용법:
    python scripts/deploy_mlops.py setup      # 전체 인프라 설정
    python scripts/deploy_mlops.py status     # 상태 확인
    python scripts/deploy_mlops.py trigger    # 수동 재학습 트리거
    python scripts/deploy_mlops.py cleanup    # 인프라 삭제

Author: HairMe ML Team
Date: 2025-12-02
"""

import os
import sys
import json
import argparse
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("❌ boto3가 설치되지 않았습니다. pip install boto3")
    sys.exit(1)

# Configuration
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
S3_BUCKET_NAME = os.getenv("MLOPS_S3_BUCKET", "hairme-mlops")
LAMBDA_FUNCTION_NAME = "hairme-model-trainer"
IAM_ROLE_NAME = "hairme-trainer-role"
EVENTBRIDGE_RULE_NAME = "hairme-weekly-retrain"

# Weekly schedule: Every Sunday at 03:00 UTC (12:00 KST)
SCHEDULE_EXPRESSION = "cron(0 3 ? * SUN *)"


def get_account_id():
    """AWS 계정 ID 조회"""
    sts = boto3.client("sts", region_name=AWS_REGION)
    return sts.get_caller_identity()["Account"]


def create_s3_bucket():
    """S3 버킷 생성"""
    print(f"📦 S3 버킷 생성 중: {S3_BUCKET_NAME}")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    try:
        # 버킷 존재 확인
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"  ✅ 버킷 이미 존재: {S3_BUCKET_NAME}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "404":
            print(f"  ❌ 버킷 확인 실패: {e}")
            return False

    try:
        # 버킷 생성
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET_NAME)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET_NAME,
                CreateBucketConfiguration={"LocationConstraint": AWS_REGION},
            )

        # 버저닝 활성화
        s3.put_bucket_versioning(
            Bucket=S3_BUCKET_NAME, VersioningConfiguration={"Status": "Enabled"}
        )

        # 초기 폴더 구조 생성
        for prefix in [
            "feedback/pending/",
            "feedback/processed/",
            "models/current/",
            "models/archive/",
            "training/logs/",
        ]:
            s3.put_object(Bucket=S3_BUCKET_NAME, Key=prefix)

        # 초기 메타데이터
        metadata = {
            "total_feedback_count": 0,
            "pending_count": 0,
            "last_training_at": None,
            "model_version": "v6",
            "created_at": datetime.utcnow().isoformat(),
        }
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key="feedback/metadata.json",
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json",
        )

        print(f"  ✅ 버킷 생성 완료: {S3_BUCKET_NAME}")
        return True

    except Exception as e:
        print(f"  ❌ 버킷 생성 실패: {e}")
        return False


def create_iam_role():
    """Lambda 실행 IAM 역할 생성"""
    print(f"🔐 IAM 역할 생성 중: {IAM_ROLE_NAME}")

    iam = boto3.client("iam", region_name=AWS_REGION)
    account_id = get_account_id()

    # Trust policy
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }

    try:
        # 역할 존재 확인
        iam.get_role(RoleName=IAM_ROLE_NAME)
        print(f"  ✅ 역할 이미 존재: {IAM_ROLE_NAME}")
        return f"arn:aws:iam::{account_id}:role/{IAM_ROLE_NAME}"
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            print(f"  ❌ 역할 확인 실패: {e}")
            return None

    try:
        # 역할 생성
        response = iam.create_role(
            RoleName=IAM_ROLE_NAME,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description="HairMe ML Trainer Lambda execution role",
        )
        role_arn = response["Role"]["Arn"]

        # 정책 연결
        policies = [
            "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess",
            "arn:aws:iam::aws:policy/AmazonDynamoDBReadOnlyAccess",
        ]

        for policy_arn in policies:
            iam.attach_role_policy(RoleName=IAM_ROLE_NAME, PolicyArn=policy_arn)

        print(f"  ✅ 역할 생성 완료: {role_arn}")

        # 역할 전파 대기
        print("  ⏳ IAM 역할 전파 대기 중 (10초)...")
        import time

        time.sleep(10)

        return role_arn

    except Exception as e:
        print(f"  ❌ 역할 생성 실패: {e}")
        return None


def create_lambda_deployment_package():
    """Lambda 배포 패키지 생성"""
    print("📦 Lambda 배포 패키지 생성 중...")

    # Lambda 핸들러 코드
    handler_code = '''
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    """
    재학습 Lambda 핸들러 (Placeholder)

    실제 학습은 EC2 Spot 또는 SageMaker에서 수행하고,
    이 Lambda는 트리거 및 모니터링 역할을 합니다.
    """
    logger.info(f"Received event: {json.dumps(event)}")

    trigger_type = event.get('trigger_type', 'unknown')

    # TODO: 실제 학습 로직 또는 EC2 Spot 인스턴스 시작

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Training triggered',
            'trigger_type': trigger_type
        })
    }
'''

    # 임시 디렉토리에 패키지 생성
    with tempfile.TemporaryDirectory() as tmpdir:
        # 핸들러 파일 생성
        handler_path = Path(tmpdir) / "lambda_function.py"
        handler_path.write_text(handler_code)

        # ZIP 파일 생성
        zip_path = Path(tmpdir) / "deployment.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(handler_path, "lambda_function.py")

        # ZIP 파일 읽기
        with open(zip_path, "rb") as f:
            zip_content = f.read()

    print(f"  ✅ 배포 패키지 생성 완료 ({len(zip_content)} bytes)")
    return zip_content


def create_lambda_function(role_arn: str):
    """Lambda 함수 생성"""
    print(f"⚡ Lambda 함수 생성 중: {LAMBDA_FUNCTION_NAME}")

    lambda_client = boto3.client("lambda", region_name=AWS_REGION)

    try:
        # 함수 존재 확인
        lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        print(f"  ✅ 함수 이미 존재: {LAMBDA_FUNCTION_NAME}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            print(f"  ❌ 함수 확인 실패: {e}")
            return False

    try:
        # 배포 패키지 생성
        zip_content = create_lambda_deployment_package()

        # 함수 생성
        response = lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime="python3.12",
            Role=role_arn,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_content},
            Description="HairMe ML Model Trainer",
            Timeout=900,  # 15분
            MemorySize=1024,  # 1GB
            Environment={
                "Variables": {
                    "MLOPS_S3_BUCKET": S3_BUCKET_NAME,
                    "AWS_REGION": AWS_REGION,
                }
            },
            Tags={"Project": "HairMe", "Component": "MLOps"},
        )

        print(f"  ✅ 함수 생성 완료: {response['FunctionArn']}")
        return True

    except Exception as e:
        print(f"  ❌ 함수 생성 실패: {e}")
        return False


def create_eventbridge_rule():
    """EventBridge 스케줄 규칙 생성"""
    print(f"📅 EventBridge 규칙 생성 중: {EVENTBRIDGE_RULE_NAME}")

    events = boto3.client("events", region_name=AWS_REGION)
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    account_id = get_account_id()

    try:
        # 규칙 생성/업데이트
        events.put_rule(
            Name=EVENTBRIDGE_RULE_NAME,
            ScheduleExpression=SCHEDULE_EXPRESSION,
            State="ENABLED",
            Description="HairMe ML 주간 재학습 스케줄 (매주 일요일 12:00 KST)",
        )

        # Lambda 타겟 설정
        lambda_arn = (
            f"arn:aws:lambda:{AWS_REGION}:{account_id}:function:{LAMBDA_FUNCTION_NAME}"
        )

        events.put_targets(
            Rule=EVENTBRIDGE_RULE_NAME,
            Targets=[
                {
                    "Id": "trainer-lambda",
                    "Arn": lambda_arn,
                    "Input": json.dumps(
                        {"trigger_type": "scheduled", "schedule": SCHEDULE_EXPRESSION}
                    ),
                }
            ],
        )

        # Lambda 권한 추가
        try:
            lambda_client.add_permission(
                FunctionName=LAMBDA_FUNCTION_NAME,
                StatementId="EventBridgeInvoke",
                Action="lambda:InvokeFunction",
                Principal="events.amazonaws.com",
                SourceArn=f"arn:aws:events:{AWS_REGION}:{account_id}:rule/{EVENTBRIDGE_RULE_NAME}",
            )
        except ClientError as e:
            if "ResourceConflictException" not in str(e):
                raise

        print(f"  ✅ 규칙 생성 완료: {SCHEDULE_EXPRESSION}")
        return True

    except Exception as e:
        print(f"  ❌ 규칙 생성 실패: {e}")
        return False


def upload_current_model():
    """현재 모델을 S3에 업로드"""
    print("📤 현재 모델 업로드 중...")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    model_path = PROJECT_ROOT / "models" / "hairstyle_recommender_v6_multitoken.pt"

    if not model_path.exists():
        print(f"  ⚠️ 모델 파일 없음: {model_path}")
        return False

    try:
        s3.upload_file(str(model_path), S3_BUCKET_NAME, "models/current/model.pt")

        # 메타데이터 업로드
        metadata = {
            "version": "v6_multitoken",
            "uploaded_at": datetime.utcnow().isoformat(),
            "source": "initial_upload",
        }
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key="models/current/metadata.json",
            Body=json.dumps(metadata, indent=2),
            ContentType="application/json",
        )

        print(f"  ✅ 모델 업로드 완료: models/current/model.pt")
        return True

    except Exception as e:
        print(f"  ❌ 모델 업로드 실패: {e}")
        return False


def setup_mlops():
    """전체 MLOps 인프라 설정"""
    print("\n" + "=" * 60)
    print("🚀 HairMe MLOps 인프라 설정 시작")
    print("=" * 60 + "\n")

    # 1. S3 버킷 생성
    if not create_s3_bucket():
        return False

    # 2. IAM 역할 생성
    role_arn = create_iam_role()
    if not role_arn:
        return False

    # 3. Lambda 함수 생성
    if not create_lambda_function(role_arn):
        return False

    # 4. EventBridge 규칙 생성
    if not create_eventbridge_rule():
        return False

    # 5. 현재 모델 업로드
    upload_current_model()

    print("\n" + "=" * 60)
    print("✅ MLOps 인프라 설정 완료!")
    print("=" * 60)
    print(
        f"""
다음 단계:
1. Lambda 환경변수에서 MLOPS_ENABLED=true 설정
2. API Lambda 재배포하여 S3 피드백 저장 활성화

환경변수 설정 (.env 또는 Lambda):
    MLOPS_ENABLED=true
    MLOPS_S3_BUCKET={S3_BUCKET_NAME}
    MLOPS_RETRAIN_THRESHOLD=100
    MLOPS_TRAINER_LAMBDA={LAMBDA_FUNCTION_NAME}

비용 예상:
    - S3 저장: $0.02/월 (1GB 미만)
    - Lambda 실행: $0 (프리티어)
    - EventBridge: $0 (무료)
    - 총 예상 비용: $0.02/월
"""
    )

    return True


def check_status():
    """MLOps 인프라 상태 확인"""
    print("\n" + "=" * 60)
    print("📊 MLOps 인프라 상태")
    print("=" * 60 + "\n")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    events = boto3.client("events", region_name=AWS_REGION)

    # S3 버킷
    print("📦 S3 버킷:")
    try:
        s3.head_bucket(Bucket=S3_BUCKET_NAME)
        print(f"  ✅ {S3_BUCKET_NAME} - 존재")

        # 피드백 메타데이터 조회
        try:
            response = s3.get_object(
                Bucket=S3_BUCKET_NAME, Key="feedback/metadata.json"
            )
            metadata = json.loads(response["Body"].read().decode("utf-8"))
            print(f"     - 총 피드백: {metadata.get('total_feedback_count', 0)}개")
            print(f"     - 대기 중: {metadata.get('pending_count', 0)}개")
            print(f"     - 마지막 학습: {metadata.get('last_training_at', 'N/A')}")
        except Exception:
            pass

    except Exception as e:
        print(f"  ❌ {S3_BUCKET_NAME} - 없음 ({e})")

    # Lambda 함수
    print("\n⚡ Lambda 함수:")
    try:
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        config = response["Configuration"]
        print(f"  ✅ {LAMBDA_FUNCTION_NAME}")
        print(f"     - 런타임: {config['Runtime']}")
        print(f"     - 메모리: {config['MemorySize']}MB")
        print(f"     - 타임아웃: {config['Timeout']}초")
        print(f"     - 마지막 수정: {config['LastModified']}")
    except Exception as e:
        print(f"  ❌ {LAMBDA_FUNCTION_NAME} - 없음")

    # EventBridge 규칙
    print("\n📅 EventBridge 규칙:")
    try:
        response = events.describe_rule(Name=EVENTBRIDGE_RULE_NAME)
        print(f"  ✅ {EVENTBRIDGE_RULE_NAME}")
        print(f"     - 상태: {response['State']}")
        print(f"     - 스케줄: {response.get('ScheduleExpression', 'N/A')}")
    except Exception:
        print(f"  ❌ {EVENTBRIDGE_RULE_NAME} - 없음")

    print()


def trigger_training():
    """수동 재학습 트리거"""
    print("\n🔄 수동 재학습 트리거 중...")

    lambda_client = boto3.client("lambda", region_name=AWS_REGION)

    try:
        response = lambda_client.invoke(
            FunctionName=LAMBDA_FUNCTION_NAME,
            InvocationType="Event",
            Payload=json.dumps(
                {
                    "trigger_type": "manual",
                    "triggered_at": datetime.utcnow().isoformat(),
                }
            ),
        )

        print(
            f"✅ 재학습 트리거 완료 (RequestId: {response['ResponseMetadata']['RequestId']})"
        )

    except Exception as e:
        print(f"❌ 트리거 실패: {e}")


def cleanup_mlops():
    """MLOps 인프라 삭제"""
    print("\n" + "=" * 60)
    print("🗑️ MLOps 인프라 삭제")
    print("=" * 60 + "\n")

    confirm = input("정말 삭제하시겠습니까? (yes/no): ")
    if confirm.lower() != "yes":
        print("취소됨")
        return

    events = boto3.client("events", region_name=AWS_REGION)
    lambda_client = boto3.client("lambda", region_name=AWS_REGION)
    iam = boto3.client("iam", region_name=AWS_REGION)
    s3 = boto3.client("s3", region_name=AWS_REGION)

    # 1. EventBridge 규칙 삭제
    print(f"📅 EventBridge 규칙 삭제: {EVENTBRIDGE_RULE_NAME}")
    try:
        events.remove_targets(Rule=EVENTBRIDGE_RULE_NAME, Ids=["trainer-lambda"])
        events.delete_rule(Name=EVENTBRIDGE_RULE_NAME)
        print("  ✅ 삭제 완료")
    except Exception as e:
        print(f"  ⚠️ {e}")

    # 2. Lambda 함수 삭제
    print(f"⚡ Lambda 함수 삭제: {LAMBDA_FUNCTION_NAME}")
    try:
        lambda_client.delete_function(FunctionName=LAMBDA_FUNCTION_NAME)
        print("  ✅ 삭제 완료")
    except Exception as e:
        print(f"  ⚠️ {e}")

    # 3. IAM 역할 삭제
    print(f"🔐 IAM 역할 삭제: {IAM_ROLE_NAME}")
    try:
        # 연결된 정책 분리
        for policy in iam.list_attached_role_policies(RoleName=IAM_ROLE_NAME)[
            "AttachedPolicies"
        ]:
            iam.detach_role_policy(
                RoleName=IAM_ROLE_NAME, PolicyArn=policy["PolicyArn"]
            )
        iam.delete_role(RoleName=IAM_ROLE_NAME)
        print("  ✅ 삭제 완료")
    except Exception as e:
        print(f"  ⚠️ {e}")

    # 4. S3 버킷은 삭제하지 않음 (데이터 보존)
    print(f"📦 S3 버킷: {S3_BUCKET_NAME} (데이터 보존을 위해 유지)")

    print("\n✅ MLOps 인프라 삭제 완료")


def main():
    parser = argparse.ArgumentParser(description="HairMe MLOps 인프라 관리")
    parser.add_argument(
        "command", choices=["setup", "status", "trigger", "cleanup"], help="실행할 명령"
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_mlops()
    elif args.command == "status":
        check_status()
    elif args.command == "trigger":
        trigger_training()
    elif args.command == "cleanup":
        cleanup_mlops()


if __name__ == "__main__":
    main()
