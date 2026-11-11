#!/usr/bin/env python3
"""
Pro 계정(미용실용 앱) 손님 기록 테이블을 만들고 TTL(expire_at)을 켠다.

테이블 (database/pro_customer_repository.py):
- 이름: hairme-pro-customers (settings.DYNAMODB_PRO_CUSTOMERS_TABLE_NAME)
- Partition Key: user_id (S, 디자이너의 user_id) / Sort Key: sk (S)
- 요금: PAY_PER_REQUEST
- TTL 속성: expire_at (epoch 초). 상담 기록은 상담일부터, 손님은 마지막 상담일부터 2년 뒤
  DynamoDB 가 지운다. 삭제가 최대 약 48시간 늦어도 서버가 읽을 때 기간을 직접 비교한다.

이미 있는 테이블은 그대로 두고, TTL 이 꺼져 있으면 켜기만 한다.
사진의 자동 삭제 규칙(pro/records/ 731일)은 scripts/apply_pro_results_lifecycle.py 가 건다.

사용법:
    python scripts/create_pro_customers_table.py            # 미리보기 (아무것도 바꾸지 않음)
    python scripts/create_pro_customers_table.py --execute  # 생성 + TTL 켜기

배포 전에 Lambda 실행 역할에 이 테이블 권한을 추가할 것: dynamodb:GetItem, PutItem,
UpdateItem, DeleteItem, Query, BatchWriteItem (트랜잭션 쓰기는 PutItem/UpdateItem/DeleteItem
권한으로 검사된다).
"""

from __future__ import annotations

import argparse
import sys

TABLE_NAME = "hairme-pro-customers"
TTL_ATTRIBUTE = "expire_at"


def table_definition(table_name: str) -> dict:
    return {
        "TableName": table_name,
        "KeySchema": [
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        "AttributeDefinitions": [
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        "BillingMode": "PAY_PER_REQUEST",
    }


def ensure_table(dynamodb, table_name: str) -> None:
    from botocore.exceptions import ClientError

    try:
        dynamodb.create_table(**table_definition(table_name))
        print(f"✅ 테이블 생성 요청: {table_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceInUseException":
            raise
        print(f"ℹ️  테이블 이미 존재: {table_name}")
    dynamodb.get_waiter("table_exists").wait(TableName=table_name)


def ensure_ttl(dynamodb, table_name: str) -> None:
    status = dynamodb.describe_time_to_live(TableName=table_name)[
        "TimeToLiveDescription"
    ]
    if (
        status.get("TimeToLiveStatus") in ("ENABLED", "ENABLING")
        and status.get("AttributeName") == TTL_ATTRIBUTE
    ):
        print(f"ℹ️  TTL 이미 켜짐: {TTL_ATTRIBUTE}")
        return
    dynamodb.update_time_to_live(
        TableName=table_name,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": TTL_ATTRIBUTE},
    )
    print(f"✅ TTL 켜기 요청: {TTL_ATTRIBUTE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--table", default=TABLE_NAME)
    parser.add_argument(
        "--execute", action="store_true", help="실제로 생성 (없으면 미리보기)"
    )
    args = parser.parse_args()

    print(f"테이블: {table_definition(args.table)}")
    print(f"TTL 속성: {TTL_ATTRIBUTE}")
    if not args.execute:
        print("\n미리보기입니다. 만들려면 --execute 를 붙이세요.")
        return 0

    import boto3

    dynamodb = boto3.client("dynamodb", region_name=args.region)
    ensure_table(dynamodb, args.table)
    ensure_ttl(dynamodb, args.table)
    print(
        "\n다음 단계:\n"
        "1. Lambda 실행 역할에 테이블 권한 추가\n"
        "2. python scripts/apply_pro_results_lifecycle.py --bucket <PHOTO_S3_BUCKET> "
        "--execute (pro/records/ 731일 규칙)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
