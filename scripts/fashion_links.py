#!/usr/bin/env python3
"""
체형 코디 쿠팡 검색 링크 관리 스크립트

배경:
    쿠팡 오픈 API(상품 검색·딥링크)는 파트너스 최종 승인(누적 수익 15만원) 전에는
    쓸 수 없다. 그래서 상품 대신 "추천 키워드 x 성별" 검색 결과 페이지마다
    파트너스 링크를 하나씩 둔다. 이 스크립트는 그 링크를 사람이 붙여넣기 쉽게
    CSV로 내보내고, 채운 CSV를 카탈로그(body_catalog.json)에 반영한다.

사용법:
    # 1) CSV 만들기 (엑셀에서 열림, 이미 넣은 링크는 그대로 채워져 나옴)
    python scripts/fashion_links.py export

    # 2) 각 줄의 search_url 을 쿠팡 파트너스 "링크 생성"에 붙여넣고,
    #    받은 https://link.coupang.com/... 링크를 affiliate_url 칸에 붙여넣는다

    # 3) 카탈로그에 반영 (링크가 들어간 줄은 active=true 가 된다)
    python scripts/fashion_links.py import
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote_plus

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data_source" / "fashion" / "body_catalog.json"
CSV_PATH = ROOT / "data_source" / "fashion" / "coupang_links.csv"

AFFILIATE_PREFIX = "https://link.coupang.com/"
PLACEHOLDER = "PLACEHOLDER"
COLUMNS = ["product_id", "gender", "keyword", "search_url", "affiliate_url"]


def search_url(query: str) -> str:
    return f"https://www.coupang.com/np/search?q={quote_plus(query)}"


def export_rows(catalog: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for link in catalog["links"]:
        url = link.get("affiliate_url", "")
        rows.append(
            {
                "product_id": link["product_id"],
                "gender": link["gender"],
                "keyword": link["keyword"],
                "search_url": search_url(link["search_query"]),
                "affiliate_url": "" if url == PLACEHOLDER else url,
            }
        )
    return rows


def apply_links(catalog: Dict[str, Any], rows: Iterable[Dict[str, str]]) -> List[str]:
    """CSV 줄을 카탈로그 links 에 반영하고 활성화된 product_id 목록을 돌려준다.

    빈 affiliate_url 은 건드리지 않는다. 형식이 틀리거나 모르는 product_id 면
    아무것도 바꾸지 않고 ValueError 를 낸다.
    """
    by_id = {link["product_id"]: link for link in catalog["links"]}
    updates = {}
    errors = []
    for row in rows:
        product_id = (row.get("product_id") or "").strip()
        url = (row.get("affiliate_url") or "").strip()
        if not url:
            continue
        if product_id not in by_id:
            errors.append(f"{product_id}: 카탈로그에 없는 product_id")
        elif not url.startswith(AFFILIATE_PREFIX):
            errors.append(f"{product_id}: {AFFILIATE_PREFIX} 로 시작하는 링크가 아님")
        else:
            updates[product_id] = url
    if errors:
        raise ValueError("\n".join(errors))

    for product_id, url in updates.items():
        by_id[product_id]["affiliate_url"] = url
        by_id[product_id]["active"] = True
    return list(updates)


def dump_catalog(obj: Any, indent: int = 0) -> str:
    """카탈로그 원래 모양대로 저장 (핏·링크 한 항목은 한 줄)"""
    pad = "  " * indent
    if isinstance(obj, dict):
        if all(not isinstance(v, (dict, list)) for v in obj.values()):
            return json.dumps(obj, ensure_ascii=False)
        body = ",\n".join(
            f"{pad}  {json.dumps(k, ensure_ascii=False)}: {dump_catalog(v, indent + 1)}"
            for k, v in obj.items()
        )
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(obj, list):
        body = ",\n".join(f"{pad}  {dump_catalog(v, indent + 1)}" for v in obj)
        return "[\n" + body + "\n" + pad + "]"
    return json.dumps(obj, ensure_ascii=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    args = parser.parse_args()

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))

    if args.command == "export":
        # utf-8-sig: 엑셀에서 한글이 깨지지 않게
        with open(args.csv, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(export_rows(catalog))
        print(f"{len(catalog['links'])}줄 저장: {args.csv}")
        return 0

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    try:
        activated = apply_links(catalog, rows)
    except ValueError as e:
        print(f"반영하지 않았습니다:\n{e}", file=sys.stderr)
        return 1
    CATALOG_PATH.write_text(dump_catalog(catalog) + "\n", encoding="utf-8")
    active = sum(1 for link in catalog["links"] if link.get("active"))
    print(f"{len(activated)}개 반영, 활성 링크 {active}/{len(catalog['links'])}개")
    return 0


if __name__ == "__main__":
    sys.exit(main())
