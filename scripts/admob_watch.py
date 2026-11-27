"""AdMob 출시 후 광고 경로 점검 (CloudWatch).

vc90(5.13.0) 프로덕션 출시 이후 "광고가 실제로 도는가"를 서버 로그로 확인한다.
AdMob 수익 금액은 AdMob 콘솔에서만 볼 수 있고, 여기서는 아래 4가지만 센다.

  - reward_ok   : /api/credits/reward-callback 200  -> 광고 시청 -> SSV -> 크레딧 지급 성공
  - reward_err  : 같은 엔드포인트 non-200          -> 서명/허용목록/중복 등으로 거부
  - paywall_402 : POST /api/v2/synthesize 402      -> 회원 크레딧 소진(광고 CTA가 떠야 할 순간)
  - paywall_429 : POST /api/v2/synthesize 429      -> 비로그인 일일 한도 소진
  - synth       : synthesis_start (로그인/비로그인) -> 전체 사용량
  - gemini      : synthesis_success 중 mode != cache -> 실제 유료 이미지 생성 횟수

왜 비용까지 세는가: 보상형 광고 1회 노출 수익(약 7원)보다 그 보상으로 돌아가는
합성 1회 비용(약 60~95원)이 10배 크다. 광고 지표를 비용과 떼어놓고 보면
"광고가 늘었다"를 좋은 신호로 착각한다. 두 숫자를 같은 화면에 둔다.

배너/전면 광고는 서버를 거치지 않아 여기 안 잡힌다. Firebase의 ad_request /
ad_shown / ad_failed (placement 파라미터) 이벤트로 확인한다.

직전 실행 결과를 state 파일에 저장해두고, 달라진 게 없으면 "변화 없음"만 출력한다.
스케줄러에서 하루 한 번 돌리는 용도.

사용:
    python scripts/admob_watch.py              # 최근 2일
    python scripts/admob_watch.py --days 7
    python scripts/admob_watch.py --days 7 --no-state   # 상태 비교 없이 집계만
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import boto3

LOG_GROUP = "/aws/lambda/hairme-analyze"
REGION = "ap-northeast-2"
STATE_PATH = os.path.join(os.path.dirname(__file__), ".admob_watch_state.json")

# Lambda 액세스 로그 한 줄 예: "GET /api/credits/reward-callback 200"
RE_REWARD = re.compile(r"/api/credits/reward-callback (\d{3})")
RE_402 = re.compile(r"POST /api/v2/synthesize 402")
RE_429 = re.compile(r"POST /api/v2/synthesize 429")
RE_AUTH_TRUE = re.compile(r'"authenticated": true')
RE_MODE_CACHE = re.compile(r'"mode": "cache"')

# 합성 1회 비용 추정 (gemini-3.1-flash-image 이미지 출력 1장).
# 출력 토큰 $60/1M 기준 1K 해상도 1120토큰 = $0.067, 0.5K 747토큰 = $0.045.
# 입력(1024px 리사이즈 사진 + 프롬프트)은 상대적으로 미미해서 무시한다.
# 보수적으로 1K 기준을 쓴다 - 과소추정보다 과대추정이 안전하다.
USD_PER_SYNTHESIS = 0.067
KRW_PER_USD = 1400

# 보상형 광고 노출 1회 수익 추정 (한국 eCPM $3~8의 중간값 $5).
USD_PER_REWARDED_IMPRESSION = 0.005

KST = timezone(timedelta(hours=9))


def _filter(client, pattern: str, start_ms: int) -> list[str]:
    """FilterLogEvents 전체 페이지를 돌며 메시지만 모은다.

    IAM에 logs:StartQuery가 없어 Insights를 못 쓴다 (운영 지표 스크립트와 동일한 제약).
    """
    messages: list[str] = []
    token = None
    while True:
        kwargs = {
            "logGroupName": LOG_GROUP,
            "startTime": start_ms,
            "filterPattern": pattern,
        }
        if token:
            kwargs["nextToken"] = token
        resp = client.filter_log_events(**kwargs)
        messages.extend(e["message"] for e in resp.get("events", []))
        token = resp.get("nextToken")
        if not token:
            return messages


def collect(days: int) -> dict:
    client = boto3.client("logs", region_name=REGION)
    start_ms = int((time.time() - days * 86400) * 1000)

    reward_codes = Counter()
    for msg in _filter(client, '"reward-callback"', start_ms):
        m = RE_REWARD.search(msg)
        if m:
            reward_codes[m.group(1)] += 1

    paywall = sum(
        1 for msg in _filter(client, '"synthesize 402"', start_ms) if RE_402.search(msg)
    )
    # 비로그인 벽(429)도 센다. 지금까지 402만 세고 있어서 "합성 요청의 2/3인
    # 비로그인 사용자가 한도에 막히는 빈도"가 완전히 안 보였다.
    paywall_anon = sum(
        1 for msg in _filter(client, '"synthesize 429"', start_ms) if RE_429.search(msg)
    )

    synth_msgs = [
        m
        for m in _filter(client, '"synthesis_start"', start_ms)
        if "synthesis_start" in m
    ]
    synth_auth = sum(1 for m in synth_msgs if RE_AUTH_TRUE.search(m))

    # 성공 합성 중 캐시 히트는 Gemini를 호출하지 않아 비용이 0이다.
    ok_msgs = [
        m
        for m in _filter(client, '"synthesis_success"', start_ms)
        if "synthesis_success" in m
    ]
    cache_hits = sum(1 for m in ok_msgs if RE_MODE_CACHE.search(m))
    gemini_calls = len(ok_msgs) - cache_hits

    return {
        "days": days,
        "checked_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        "reward_ok": reward_codes.get("200", 0),
        "reward_err": sum(v for k, v in reward_codes.items() if k != "200"),
        "reward_codes": dict(reward_codes),
        "paywall_402": paywall,
        "paywall_429": paywall_anon,
        "synth_total": len(synth_msgs),
        "synth_auth": synth_auth,
        "synth_anon": len(synth_msgs) - synth_auth,
        "synth_ok": len(ok_msgs),
        "cache_hits": cache_hits,
        "gemini_calls": gemini_calls,
    }


def render(cur: dict, prev: dict | None) -> tuple[str, bool]:
    """사람이 읽을 보고문과 '보고할 만한 변화가 있는지' 여부를 돌려준다."""
    days = max(cur["days"], 1)
    gemini = cur.get("gemini_calls", 0)
    # 월 환산 비용: 측정 기간의 일평균 Gemini 호출 x 30일 x 장당 단가
    cost_krw_month = (gemini / days) * 30 * USD_PER_SYNTHESIS * KRW_PER_USD
    # 보상형이 벌어들인 추정 수익 (지급 성공 1건 = 노출 1회로 본다)
    reward_krw = cur["reward_ok"] * USD_PER_REWARDED_IMPRESSION * KRW_PER_USD
    # 그 보상으로 돌아간 합성 비용
    reward_cost_krw = cur["reward_ok"] * USD_PER_SYNTHESIS * KRW_PER_USD

    lines = [
        f"[HairMe 광고 점검] {cur['checked_at']} / 최근 {cur['days']}일",
        f"  보상 지급 성공(200)  : {cur['reward_ok']}",
        f"  보상 콜백 거부       : {cur['reward_err']}"
        + (f" {cur['reward_codes']}" if cur["reward_err"] else ""),
        f"  크레딧 소진 402      : {cur['paywall_402']} (회원)",
        f"  일일 한도 429        : {cur.get('paywall_429', 0)} (비로그인)",
        f"  합성 요청            : {cur['synth_total']} (로그인 {cur['synth_auth']} / 비로그인 {cur['synth_anon']})",
        f"  합성 성공            : {cur.get('synth_ok', 0)} (캐시 {cur.get('cache_hits', 0)} / Gemini 유료 {gemini})",
        f"  Gemini 비용 월환산   : 약 {cost_krw_month:,.0f}원",
    ]

    if cur["reward_ok"]:
        lines.append(
            f"  보상형 손익          : 수익 약 {reward_krw:,.0f}원 vs 합성 비용 약 {reward_cost_krw:,.0f}원"
            f" (회수율 {reward_krw / reward_cost_krw * 100:.0f}%)"
        )

    notable = False
    if prev is None:
        notable = True
        lines.append("  * 첫 실행 — 기준선을 기록했습니다.")
    else:
        if cur["reward_ok"] > 0 and prev.get("reward_ok", 0) == 0:
            notable = True
            lines.append(
                "  *** 첫 광고 보상 지급 확인! 광고→AdMob→SSV→크레딧 경로가 살아있습니다. ***"
            )
        elif cur["reward_ok"] != prev.get("reward_ok"):
            notable = True
            lines.append(
                f"  * 보상 지급 변화: {prev.get('reward_ok')} -> {cur['reward_ok']}"
            )
        if cur["reward_err"] > prev.get("reward_err", 0):
            notable = True
            lines.append(
                f"  * 보상 콜백 거부 증가: {prev.get('reward_err')} -> {cur['reward_err']}"
                " (서명/허용목록/일일상한 확인 필요)"
            )
        if cur.get("paywall_429", 0) > prev.get("paywall_429", 0):
            notable = True
            lines.append(
                f"  * 비로그인 일일 한도 거절 증가: {prev.get('paywall_429', 0)} -> {cur.get('paywall_429', 0)}"
                " (비로그인은 보상형이 닿지 않는 세그먼트다 - 배너/전면만 도달 가능)"
            )
        if cur["reward_ok"] == 0 and cur["paywall_402"] > prev.get("paywall_402", 0):
            notable = True
            lines.append(
                "  * 크레딧 벽에 부딪힌 사용자는 늘었는데 광고 시청은 0 —"
                " 광고 CTA가 안 보이거나 광고가 채워지지 않는 중일 수 있습니다."
            )
        if not notable:
            lines.append("  변화 없음.")

    return "\n".join(lines), notable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=2)
    ap.add_argument(
        "--no-state", action="store_true", help="직전 결과와 비교하지 않는다"
    )
    args = ap.parse_args()

    cur = collect(args.days)

    prev = None
    if not args.no_state and os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                prev = json.load(f)
        except (OSError, json.JSONDecodeError):
            prev = None  # 손상됐으면 기준선을 새로 잡는다

    report, notable = render(cur, prev)
    print(report)

    if not args.no_state:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=2)

    # 스케줄러에서 "변화 있을 때만" 걸러내기 쉽도록 종료 코드로 알린다.
    return 10 if notable else 0


if __name__ == "__main__":
    sys.exit(main())
