#!/usr/bin/env python3
"""
운영 사진 버킷에 Pro 계정(미용실용 앱) 사진의 자동 삭제 규칙을 추가한다.

1. pro/results/ - 7일 (expire-pro-results)
   Pro 계정은 손님 사진을 다루므로 합성 결과를 pro/results/{user_id}/ 에 따로 저장하고
   짧게 보관한다 (services/photo_storage_service.py). 손님용 results/ 규칙(180일)과
   접두사가 겹치지 않는다. 손님에게 보내는 짧은 주소의 연결 정보(pro/results/_share/)도
   같은 접두사 아래라 이 규칙 하나로 함께 지워진다.

2. pro/records/ - 731일 (expire-pro-records)
   디자이너가 보관하는 손님 기록의 상담 사진 (services/pro_records_service.py).
   기록은 상담일부터 2년(730일, settings.PRO_RECORD_RETENTION_DAYS) 보관하고 DynamoDB TTL 이
   지운다. 사진은 상담 뒤에 올라오므로 하루 더 긴 731일로 두어 기록보다 먼저 지워지지 않게 한다.
   서버는 기간이 지난 기록을 읽을 때 빼므로 그 사이 사진이 보이는 일은 없다.

put_bucket_lifecycle_configuration 은 규칙 전체를 덮어쓰므로, 기존 규칙(cache/, results/ 등)을
읽어서 위 규칙만 추가·갱신한 뒤 다시 쓴다 (scripts/apply_body_photo_lifecycle.py 와 같은 방식).

사용법:
    python scripts/apply_pro_results_lifecycle.py --bucket <PHOTO_S3_BUCKET>            # 미리보기
    python scripts/apply_pro_results_lifecycle.py --bucket <PHOTO_S3_BUCKET> --execute  # 적용
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

RULE_ID = "expire-pro-results"
PREFIX = "pro/results/"
DAYS = 7

RECORDS_RULE_ID = "expire-pro-records"
RECORDS_PREFIX = "pro/records/"
RECORDS_DAYS = 731

RULES: List[Dict[str, Any]] = [
    {
        "ID": RULE_ID,
        "Status": "Enabled",
        "Filter": {"Prefix": PREFIX},
        "Expiration": {"Days": DAYS},
    },
    {
        "ID": RECORDS_RULE_ID,
        "Status": "Enabled",
        "Filter": {"Prefix": RECORDS_PREFIX},
        "Expiration": {"Days": RECORDS_DAYS},
    },
]


def merged_rules(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """기존 규칙은 그대로 두고 이 스크립트의 규칙만 추가·갱신한다"""
    ours = {rule["ID"] for rule in RULES}
    rules = [r for r in existing if r.get("ID") not in ours]
    rules.extend(dict(rule) for rule in RULES)
    return rules


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument(
        "--execute", action="store_true", help="실제로 적용 (없으면 미리보기)"
    )
    args = parser.parse_args()

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", region_name=args.region)
    try:
        existing = s3.get_bucket_lifecycle_configuration(Bucket=args.bucket)["Rules"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchLifecycleConfiguration":
            raise
        existing = []

    rules = merged_rules(existing)
    print(json.dumps(rules, ensure_ascii=False, indent=2))
    if not args.execute:
        print("\n미리보기입니다. 적용하려면 --execute 를 붙이세요.")
        return 0
    s3.put_bucket_lifecycle_configuration(
        Bucket=args.bucket, LifecycleConfiguration={"Rules": rules}
    )
    print(
        f"\n✅ {args.bucket}: {PREFIX} {DAYS}일, "
        f"{RECORDS_PREFIX} {RECORDS_DAYS}일 자동 삭제 규칙 적용"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
