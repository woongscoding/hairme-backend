#!/usr/bin/env python3
"""
운영 사진 버킷에 Pro 계정(미용실용 앱) 결과 7일 자동 삭제 규칙을 추가한다.

Pro 계정은 손님 사진을 다루므로 결과를 pro/results/{user_id}/ 에 따로 저장하고
짧게 보관한다 (services/photo_storage_service.py). 손님용 results/ 규칙(180일)과
접두사가 겹치지 않는다.

put_bucket_lifecycle_configuration 은 규칙 전체를 덮어쓰므로, 기존 규칙(cache/, results/ 등)을
읽어서 pro/results/ 규칙만 추가·갱신한 뒤 다시 쓴다 (scripts/apply_body_photo_lifecycle.py 와 같은 방식).

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


def merged_rules(existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rules = [r for r in existing if r.get("ID") != RULE_ID]
    rules.append(
        {
            "ID": RULE_ID,
            "Status": "Enabled",
            "Filter": {"Prefix": PREFIX},
            "Expiration": {"Days": DAYS},
        }
    )
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
    print(f"\n✅ {args.bucket}: {PREFIX} {DAYS}일 자동 삭제 규칙 적용")
    return 0


if __name__ == "__main__":
    sys.exit(main())
