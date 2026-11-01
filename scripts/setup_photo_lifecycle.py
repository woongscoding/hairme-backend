"""사진 버킷 수명주기(파기 기한) 설정 - 수익화 v2.1 Phase 0

training/ (학습 동의분)과 reports/ (AI 이미지 신고)에 만료 규칙을 건다.
보관 기간은 개인정보처리방침에 적는 값과 같아야 하므로 기본값을 두지 않는다.

put_bucket_lifecycle_configuration 은 규칙 전체를 "치환"한다. 그래서 기존 규칙
(expire-cache, expire-results 등)을 읽어 같은 ID 만 바꾸고 나머지는 그대로 둔다.

기본은 dry-run (바뀔 규칙만 출력). 실제 적용은 --apply.

    python scripts/setup_photo_lifecycle.py --training-days 365 --reports-days 365
    python scripts/setup_photo_lifecycle.py --training-days 365 --reports-days 365 --apply

주의: 적용하면 기한이 지난 기존 객체도 S3 가 하루 안에 비동기로 삭제한다 (되돌릴 수 없음).
"""

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

DEFAULT_BUCKET = "hairme-photos-364042451408"


def build_rules(training_days: int, reports_days: int) -> list:
    return [
        {
            "ID": "expire-training",
            "Status": "Enabled",
            "Filter": {"Prefix": "training/"},
            "Expiration": {"Days": training_days},
        },
        {
            "ID": "expire-reports",
            "Status": "Enabled",
            "Filter": {"Prefix": "reports/"},
            "Expiration": {"Days": reports_days},
        },
    ]


def merge_rules(existing: list, new_rules: list) -> list:
    """같은 ID 는 교체, 나머지 기존 규칙은 유지"""
    new_ids = {rule["ID"] for rule in new_rules}
    return [rule for rule in existing if rule.get("ID") not in new_ids] + new_rules


def current_rules(s3, bucket: str) -> list:
    try:
        return s3.get_bucket_lifecycle_configuration(Bucket=bucket).get("Rules", [])
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchLifecycleConfiguration":
            return []
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--training-days", type=int, required=True)
    parser.add_argument("--reports-days", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="실제로 적용")
    args = parser.parse_args()

    if args.training_days < 1 or args.reports_days < 1:
        parser.error("보관 기간은 1일 이상이어야 합니다")

    s3 = boto3.client("s3", region_name=args.region)
    existing = current_rules(s3, args.bucket)
    merged = merge_rules(existing, build_rules(args.training_days, args.reports_days))

    print(f"버킷: {args.bucket}")
    print("현재 규칙:")
    print(json.dumps(existing, indent=2, ensure_ascii=False, default=str))
    print("적용할 규칙:")
    print(json.dumps(merged, indent=2, ensure_ascii=False, default=str))

    if not args.apply:
        print("\n(dry-run) 적용하려면 --apply 를 붙이세요.")
        return 0

    s3.put_bucket_lifecycle_configuration(
        Bucket=args.bucket, LifecycleConfiguration={"Rules": merged}
    )
    print("\n✅ 적용 완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
