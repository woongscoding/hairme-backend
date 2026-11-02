#!/usr/bin/env python3
"""스타일 반응률 집계 -> s3://hairme-mlops/engagement/current.json

추천 반응률 보너스(services/rec_engagement.py)가 읽는 JSON 을 만든다.
집계 규칙은 services/mlops/engagement.py 머리말 참고.

    노출: hairme-analysis 최근 N일(기본 60) 분석의 ML 카드 (Scan, 읽기 전용)
    반응: s3://hairme-mlops/feedback/pending/** + feedback/processed/** 의 npz
          (암묵 신호 share/save/synthesize/shown_not_chosen + 명시 good/bad)

기본은 dry-run: 로컬 파일로만 쓰고 요약을 출력한다. --write 일 때만
engagement/current.json 과 engagement/archive/{YYYY-MM-DD}.json 을 올린다.
--write 전 가드: npz 읽기 오류 비율 > --max-read-error-ratio(5%), 노출 <
--min-exposures, mu <= 0, styles 비어 있음 중 하나라도 걸리면 업로드하지 않고
종료 코드 2 로 끝난다 (서빙은 이전 테이블을 신선도 한도까지 계속 쓴다).

필요 권한 (GitHub Actions 사용자 / 로컬 실행자):
    dynamodb:Scan                          on table/hairme-analysis
    s3:ListBucket                          on hairme-mlops (prefix feedback/)
    s3:GetObject                           on hairme-mlops/feedback/*
    s3:PutObject (--write 일 때만)          on hairme-mlops/engagement/*

기본 파라미터 근거 (2026-11-02 시뮬레이션: 운영 모델로 재현한 9/17~ 분석 849건의 점수 풀
+ 가상 반응, 월 분석 640 / 합성 500, 합성 중 저장 30% / 공유 10%, 매일 60일 창
재집계, 시드 8개 평균)
    a1 = a2 = ag = 20 (반응 가중치 단위, m = a / mu). mu ~ 0.5 에서 m ~ 40 노출.
        a=40 은 효과 없을 때 잡음 보너스가 0.69 -> 0.61 점으로 조금 줄지만 스타일별
        효과 차이(sd 0.3) 세계의 기대 반응 개선이 +4.0% -> +2.8% 로 줄고 탐지가 느려진다.
        mu 에 비례하는 사전이라 명시 피드백만 있는 지금(mu 0.012)도 좋아요 몇 개로
        보너스가 상한에 붙지 않는다 (실데이터 최대 |1.09| 점).
    BETA = 3, CAP = 1.5 (서빙, config/settings.py): CAP 1.5 와 3 은 탐지 속도·기대
        반응이 거의 같았다 (BETA 3 에서 상한에 닿는 셀이 드묾). 1.5 = 동점권 폭/2 라서
        보너스만으로 원점수 3점 넘게 차이 나는 두 스타일의 순서는 바뀌지 않는다.
    share = save = 3: 앱은 공유 시트를 연 시점(취소 포함)에 share 를 보낸다.
    위치 보정 기본 꺼짐: 탐색이 꺼진 지금은 같은 스타일이 여러 순위에 나오는 일이
        드물어 위치·품질 효과를 나눠 추정할 수 없다. --position-adjust 로만 켠다.

사용법:
    python scripts/build_style_engagement.py                 # dry-run (읽기만)
    python scripts/build_style_engagement.py --days 30 --no-explicit
    python scripts/build_style_engagement.py --write         # S3 업로드 (운영, 가드 통과 시만)
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
# image_hash / user_id 는 주체당 기여 상한에만 쓰고 해시로만 메모리에 남긴다 (출력 없음)
PROJECTION_FIELDS = (
    "analysis_id",
    "created_at",
    "gender",
    "face_shape",
    "recommended_styles",
    "recommendations",
    "image_hash",
    "user_id",
)
EXIT_GUARD_FAILED = 2


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
    parser.add_argument("--ag", type=float, default=eng.DEFAULT_AG)
    parser.add_argument(
        "--position-prior", type=float, default=eng.DEFAULT_POSITION_PRIOR
    )
    parser.add_argument(
        "--position-adjust",
        action="store_true",
        help="위치 보정 켜기 (여러 순위에 노출된 스타일로만 추정, 기본 꺼짐)",
    )
    parser.add_argument("--min-exposures", type=int, default=500)
    parser.add_argument("--max-read-error-ratio", type=float, default=0.05)
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


# ========== 가드 ==========
def write_guard(
    payload: Dict[str, Any],
    *,
    read_errors: int,
    keys_total: int,
    min_exposures: int,
    max_read_error_ratio: float,
) -> List[str]:
    """업로드하면 안 되는 이유 목록 (비어 있으면 통과)"""
    problems: List[str] = []
    if keys_total > 0 and read_errors / keys_total > max_read_error_ratio:
        problems.append(
            f"npz 읽기 오류 {read_errors}/{keys_total} > {max_read_error_ratio:.0%}"
        )
    exposures = int(payload.get("summary", {}).get("exposures", 0))
    if exposures < min_exposures:
        problems.append(f"노출 {exposures} < {min_exposures}")
    if not payload.get("mu") or payload["mu"] <= 0:
        problems.append(f"mu {payload.get('mu')} <= 0")
    if not payload.get("styles"):
        problems.append("styles 비어 있음")
    return problems


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
        base = totals.get("mean_rate") or mu
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
                "mean_rate": totals.get("mean_rate"),
                "top_lift": (
                    round((best[1]["rate"] - base) / base, 3)
                    if best and base > 0
                    else None
                ),
                "max_save_share_subjects": max(
                    (e.get("subjects", 0) for e in entries.values()), default=0
                ),
                "max_save_share_auth_subjects": max(
                    (e.get("auth_subjects", 0) for e in entries.values()), default=0
                ),
            }
        )
    return {
        "version": payload["version"],
        "window": payload["window"],
        "mu": mu,
        "m1": payload["params"]["m1"],
        "m2": payload["params"]["m2"],
        "mg": payload["params"]["mg"],
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
        ag=args.ag,
        position_adjust=args.position_adjust,
        position_prior=args.position_prior,
        face_shape_since=args.face_shape_since,
    )

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(body, encoding="utf-8")

    summary = summarize(payload, exposure_stats, samples, sample_stats, len(keys))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"로컬 출력: {args.output} ({len(body):,} bytes)")

    problems = write_guard(
        payload,
        read_errors=sample_stats.get("read_error", 0),
        keys_total=len(keys),
        min_exposures=args.min_exposures,
        max_read_error_ratio=args.max_read_error_ratio,
    )
    for problem in problems:
        logger.error(f"가드: {problem}")

    if not args.write:
        logger.info("dry-run - S3 에 쓰지 않음 (--write 로 업로드)")
        return 0

    if problems:
        logger.error("가드 실패 - S3 에 쓰지 않음")
        return EXIT_GUARD_FAILED

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
