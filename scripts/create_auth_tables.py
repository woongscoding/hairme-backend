"""회원/크레딧 시스템용 AWS 리소스 생성 스크립트

생성 리소스:
1. DynamoDB hairme-users        (PK: user_id, GSI: kakao_id-index)
2. DynamoDB hairme-credit-ledger (PK: user_id, SK: sk)
3. (선택) S3 사진 버킷 (--create-bucket <이름>)

사용법:
    python scripts/create_auth_tables.py --region ap-northeast-2
    python scripts/create_auth_tables.py --create-bucket hairme-photos
"""

import argparse
import sys

import boto3
from botocore.exceptions import ClientError


def create_users_table(dynamodb, table_name: str) -> None:
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "user_id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "kakao_id", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "kakao_id-index",
                    "KeySchema": [{"AttributeName": "kakao_id", "KeyType": "HASH"}],
                    "Projection": {"ProjectionType": "ALL"},
                }
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"✅ 테이블 생성 요청: {table_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"ℹ️  테이블 이미 존재: {table_name}")
        else:
            raise


def create_ledger_table(dynamodb, table_name: str) -> None:
    try:
        dynamodb.create_table(
            TableName=table_name,
            KeySchema=[
                {"AttributeName": "user_id", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "user_id", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        print(f"✅ 테이블 생성 요청: {table_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ResourceInUseException":
            print(f"ℹ️  테이블 이미 존재: {table_name}")
        else:
            raise


def create_photo_bucket(s3, bucket_name: str, region: str) -> None:
    try:
        s3.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
        print(f"✅ S3 버킷 생성: {bucket_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] in (
            "BucketAlreadyOwnedByYou",
            "BucketAlreadyExists",
        ):
            print(f"ℹ️  버킷 이미 존재: {bucket_name}")
        else:
            raise

    # 퍼블릭 액세스 전면 차단 (presigned URL로만 접근)
    s3.put_public_access_block(
        Bucket=bucket_name,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )

    # 캐시는 90일, 결과물은 180일 후 자동 삭제 (원본은 학습 데이터라 무기한 보관)
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket_name,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "expire-cache",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "cache/"},
                    "Expiration": {"Days": 90},
                },
                {
                    "ID": "expire-results",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "results/"},
                    "Expiration": {"Days": 180},
                },
            ]
        },
    )
    print("✅ 퍼블릭 차단 + 수명주기 정책 설정 완료")


def main() -> int:
    parser = argparse.ArgumentParser(description="회원/크레딧 AWS 리소스 생성")
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--users-table", default="hairme-users")
    parser.add_argument("--ledger-table", default="hairme-credit-ledger")
    parser.add_argument(
        "--create-bucket",
        metavar="BUCKET_NAME",
        help="사진 저장용 S3 버킷도 생성 (예: hairme-photos)",
    )
    args = parser.parse_args()

    dynamodb = boto3.client("dynamodb", region_name=args.region)
    create_users_table(dynamodb, args.users_table)
    create_ledger_table(dynamodb, args.ledger_table)

    if args.create_bucket:
        s3 = boto3.client("s3", region_name=args.region)
        create_photo_bucket(s3, args.create_bucket, args.region)
        print(f"\n환경변수 설정 필요: PHOTO_S3_BUCKET={args.create_bucket}")

    print(
        "\n다음 단계:\n"
        '1. JWT 시크릿 생성: python -c "import secrets; print(secrets.token_urlsafe(64))"\n'
        "2. 로컬: .env에 JWT_SECRET_KEY 추가\n"
        "3. 프로덕션: Secrets Manager에 hairme-jwt-secret 등록\n"
        "   aws secretsmanager create-secret --name hairme-jwt-secret --secret-string '<시크릿>'\n"
        "4. Lambda IAM 역할에 새 테이블/버킷 권한 추가"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
