#!/usr/bin/env python3
"""스타일 반응률 집계 -> s3://hairme-mlops/engagement/current.json

추천 반응률 보너스(services/rec_engagement.py)가 읽는 JSON 을 만든다.
집계 규칙은 services/mlops/engagement.py 머리말 참고.

    노출: hairme-analysis 최근 N일(기본 60) 분석의 ML 카드 (Scan, 읽기 전용)
    반응: s3://hairme-mlops/feedback/pending/** + feedback/processed/** 의 npz
          (암묵 신호 share/save/synthesize/shown_not_chosen + 명시 good/bad)

기본은 dry-run: 로컬 파일로만 쓰고 요약을 출력한다. --write 일 때만
engagement/current.json 과 engagement/archive/{YYYY-MM-DD}.json 을 올린다.

필요 권한 (GitHub Actions 사용자 / 로컬 실행자):
    dynamodb:Scan                          on table/hairme-analysis
    s3:ListBucket                          on hairme-mlops (prefix feedback/)
    s3:GetObject                           on hairme-mlops/feedback/*
    s3:PutObject (--write 일 때만)          on hairme-mlops/engagement/*

기본 파라미터 근거 (2026-11-02 시뮬레이션: 운영 모델로 재현한 9/17~ 분석 849건의 점수 풀
+ 가상 반응, 월 분석 640 / 합성 500, 매일 60일 창 재집계, 시드 5개 평균)
    a1 = a2 = 20 (반응 가중치 단위, m = a / mu). mu ~ 0.5 에서 m ~ 40 노출.
        a=5~10 은 반응 2배 스타일을 1주 빨리 잡지만 효과가 없을 때 잡음 보너스가
        평균 0.5 / p90 1.0 점으로 커진다. a=40 은 잡음 0.3 이지만 작은 그룹이 6주 이상.
        a=20, BETA=3 에서 잡음 보너스 평균 0.40 / p90 0.76 점 (상한 3의 1/4 수준).
        mu 에 비례하는 사전이라 명시 피드백만 있는 지금(mu 0.01)도 좋아요 몇 개로
        보너스가 상한에 붙지 않는다 (실데이터 최대 1.15 점).
    BETA = 3, CAP = 3 (서빙, config/settings.py): lift +1(반응 2배)을 상한 3점에 대응.
        CAP 은 탐색 동점권 폭 REC_EXPLORE_MARGIN(3점)과 같게 해서 보너스만으로
        동점권을 넘는 순위 역전은 하지 않는다.
    위치 보정 사전 강도 20: 1위/3위 반응 차이가 실제로 있을 때 수백 건이면 반영.

사용법:
    python scripts/build_style_engagement.py                 # dry-run (읽기만)
    python scripts/build_style_engagement.py --days 30 --no-explicit
    python scripts/build_style_engagement.py --write         # S3 업로드 (운영)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.mlops import engagement as eng  # noqa: E402

logger = logging.getLogger("build_style_engagement")

DEFAULT_TABLE = "hairme-analysis"
DEFAULT_BUCKET = "hairme-mlops"
DEFAULT_REGION = "ap-northeast-2"
FEEDBACK_PREFIXES = ("feedback/pending/", "feedback/processed/")
CURRENT_KEY = "engagement/current.json"
ARCHIVE_PREFIX = "engagement/archive/"

# 노출 집계에 필요한 필드만 읽는다 (user_id / device_id 는 읽지 않는다)
PROJECTION_FIELDS = (
    "analysis_id",
    "created_at",
    "gender",
    "face_shape",
    "recommended_styles",
    "recommendations",
)


# ========== 인자 ==========
def parse_weights(text: Optional[str]) -> Dict[str, float]:
    """ "save=3,share=4" -> 기본값에 덮어쓴 가중치"""
    weights = dict(eng.DEFAULT_WEIGHTS)
    if not text:
        return weights
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in weights:
            raise argparse.ArgumentTypeError(f"unknown signal: {name}")
        weights[name] = float(value)
    return weights


def parse_date(text: str) -> datetime:
    parsed = eng.parse_timestamp(text if "T" in text else f"{text}T00:00:00+00:00")
    if parsed is None:
        raise argparse.ArgumentTypeError(f"invalid date: {text}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--days", type=int, default=eng.DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--until", type=parse_date, default=None, help="기본: 지금 (UTC)"
    )
    parser.add_argument(
        "--face-shape-since",
        type=parse_date,
        default=parse_date(eng.FACE_SHAPE_RELIABLE_SINCE),
        help="이 시각 이전 분석은 그룹 통계에서 제외 (얼굴형 재보정 이전)",
    )
    parser.add_argument("--weights", type=parse_weights, default=parse_weights(None))
    parser.add_argument("--no-explicit", action="store_true", help="명시 피드백 제외")
    parser.add_argument("--a1", type=float, default=eng.DEFAULT_A1)
    parser.add_argument("--a2", type=float, default=eng.DEFAULT_A2)
    parser.add_argument(
        "--position-prior", type=float, default=eng.DEFAULT_POSITION_PRIOR
    )
    parser.add_argument("--no-position-adjust", action="store_true")
    parser.add_argument(
        "--table", default=os.getenv("DYNAMODB_TABLE_NAME", DEFAULT_TABLE)
    )
    parser.add_argument(
        "--bucket", default=os.getenv("MLOPS_S3_BUCKET", DEFAULT_BUCKET)
    )
    parser.add_argument("--region", default=os.getenv("AWS_REGION", DEFAULT_REGION))
    parser.add_argument(
        "--output",
        default=os.path.join(tempfile.gettempdir(), "hairme_engagement_current.json"),
        help="로컬 출력 경로 (dry-run 도 항상 쓴다)",
    )
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument(
        "--write", action="store_true", help="S3 current.json + 날짜별 아카이브 업로드"
    )
    return parser


# ========== 읽기 ==========
def scan_analyses(table, since: datetime) -> Iterator[Dict[str, Any]]:
    """created_at >= since 인 분석 (Scan + 필터, 읽기 전용)"""
    names = {f"#f{i}": name for i, name in enumerate(PROJECTION_FIELDS)}
    kwargs: Dict[str, Any] = {
        "ProjectionExpression": ", ".join(names),
        "ExpressionAttributeNames": names,
        "FilterExpression": "#f1 >= :since",
        "ExpressionAttributeValues": {":since": since.strftime("%Y-%m-%d")},
    }
    while True:
        response = table.scan(**kwargs)
        yield from response.get("Items", [])
        last = response.get("LastEvaluatedKey")
        if not last:
            return
        kwargs["ExclusiveStartKey"] = last


def list_feedback_keys(s3, bucket: str) -> List[str]:
    keys: List[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for prefix in FEEDBACK_PREFIXES:
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                key = obj.get("Key", "")
                if key.endswith(".npz"):
                    keys.append(key)
    return keys


def load_samples(
    s3, bucket: str, keys: List[str], max_workers: int
) -> Tuple[List[eng.ReactionSample], Counter]:
    """npz metadata -> 반응 샘플. 실패·해석 불가는 개수만 센다"""
    stats: Counter = Counter()

    def fetch(key: str) -> Optional[eng.ReactionSample]:
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        except Exception as e:
            logger.warning(f"npz 읽기 실패 {key}: {e}")
            stats["read_error"] += 1
            return None
        metadata = eng.read_npz_metadata(body)
        if metadata is None:
            stats["no_metadata"] += 1
            return None
        sample = eng.sample_from_metadata(metadata)
        if sample is None:
            stats["unparsed"] += 1
        return sample

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        results = list(pool.map(fetch, keys))
    samples = [s for s in results if s is not None]
    return samples, stats


# ========== 요약 ==========
def summarize(
    payload: Dict[str, Any],
    exposure_stats: eng.ExposureStats,
    samples: List[eng.ReactionSample],
    sample_stats: Counter,
    keys_total: int,
) -> Dict[str, Any]:
    mu = payload["mu"]
    groups = []
    for gkey, totals in sorted(
        payload["group_totals"].items(), key=lambda kv: -kv[1]["exposures"]
    ):
        entries = payload["groups"][gkey]
        best = max(entries.items(), key=lambda kv: kv[1]["rate"], default=None)
        groups.append(
            {
                "group": gkey,
                "exposures": totals["exposures"],
                "reaction_weight": totals["reactions"],
                "styles": totals["styles"],
                "styles_with_30plus_exposures": sum(
                    1 for e in entries.values() if e["exposures"] >= 30
                ),
                "top_style": best[0] if best else None,
                "top_lift": (
                    round((best[1]["rate"] - mu) / mu, 3) if best and mu > 0 else None
                ),
            }
        )
    return {
        "version": payload["version"],
        "window": payload["window"],
        "mu": mu,
        "m1": payload["params"]["m1"],
        "m2": payload["params"]["m2"],
        "position_factors": payload["params"]["position_factors"],
        "exposures": exposure_stats.as_dict(),
        "npz_objects": keys_total,
        "samples_by_kind_all": dict(Counter(s.kind for s in samples)),
        "sample_read_issues": dict(sample_stats),
        "reactions": payload["summary"],
        "styles_with_exposure": len(payload["styles"]),
        "groups": groups,
    }


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)

    import boto3
    from botocore.config import Config

    until = args.until or datetime.now(timezone.utc)
    since = until - timedelta(days=args.days)
    include_explicit = not args.no_explicit

    table = boto3.resource("dynamodb", region_name=args.region).Table(args.table)
    s3 = boto3.client(
        "s3",
        region_name=args.region,
        config=Config(max_pool_connections=max(10, args.max_workers)),
    )

    logger.info(f"분석 스캔: {args.table} {since.isoformat()} ~ {until.isoformat()}")
    exposures, exposure_stats = eng.collect_exposures(
        scan_analyses(table, since), since, until, args.face_shape_since
    )
    logger.info(f"ML 노출 {len(exposures)}건 / 분석 {exposure_stats.analyses}건")

    keys = list_feedback_keys(s3, args.bucket)
    logger.info(f"피드백 npz {len(keys)}개 읽는 중")
    samples, sample_stats = load_samples(s3, args.bucket, keys, args.max_workers)

    reactions = eng.reduce_reactions(samples, args.weights, include_explicit)
    payload = eng.build_payload(
        exposures,
        reactions,
        since=since,
        until=until,
        weights=args.weights,
        include_explicit=include_explicit,
        a1=args.a1,
        a2=args.a2,
        position_adjust=not args.no_position_adjust,
        position_prior=args.position_prior,
        face_shape_since=args.face_shape_since,
    )

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(body, encoding="utf-8")

    summary = summarize(payload, exposure_stats, samples, sample_stats, len(keys))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"로컬 출력: {args.output} ({len(body):,} bytes)")

    if not args.write:
        logger.info("dry-run - S3 에 쓰지 않음 (--write 로 업로드)")
        return 0

    archive_key = f"{ARCHIVE_PREFIX}{until.strftime('%Y-%m-%d')}.json"
    for key in (archive_key, CURRENT_KEY):
        s3.put_object(
            Bucket=args.bucket,
            Key=key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(f"업로드: s3://{args.bucket}/{key}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
