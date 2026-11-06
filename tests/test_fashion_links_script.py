"""scripts/fashion_links.py: 쿠팡 검색 링크 CSV 내보내기/반영"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fashion_links.py"
_spec = importlib.util.spec_from_file_location("fashion_links", SCRIPT)
fashion_links = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fashion_links)

CATALOG = json.loads(fashion_links.CATALOG_PATH.read_text(encoding="utf-8"))


def _catalog():
    return copy.deepcopy(CATALOG)


def test_export_has_search_url_and_blank_placeholder():
    rows = fashion_links.export_rows(_catalog())
    assert len(rows) == len(CATALOG["links"])
    first = rows[0]
    assert first["product_id"] == "fashion-m-01"
    assert first["search_url"].startswith("https://www.coupang.com/np/search?q=")
    assert first["affiliate_url"] == ""


def test_apply_activates_filled_rows_only():
    catalog = _catalog()
    rows = [
        {
            "product_id": "fashion-m-01",
            "affiliate_url": " https://link.coupang.com/a/x ",
        },
        {"product_id": "fashion-m-02", "affiliate_url": ""},
    ]
    assert fashion_links.apply_links(catalog, rows) == ["fashion-m-01"]
    by_id = {link["product_id"]: link for link in catalog["links"]}
    assert by_id["fashion-m-01"]["affiliate_url"] == "https://link.coupang.com/a/x"
    assert by_id["fashion-m-01"]["active"] is True
    assert by_id["fashion-m-02"]["active"] is False


@pytest.mark.parametrize(
    "row",
    [
        {"product_id": "fashion-m-01", "affiliate_url": "https://www.coupang.com/np/x"},
        {
            "product_id": "fashion-zz-99",
            "affiliate_url": "https://link.coupang.com/a/x",
        },
    ],
)
def test_apply_rejects_bad_rows_without_partial_changes(row):
    catalog = _catalog()
    good = {
        "product_id": "fashion-m-02",
        "affiliate_url": "https://link.coupang.com/a/y",
    }
    with pytest.raises(ValueError):
        fashion_links.apply_links(catalog, [good, row])
    assert catalog == CATALOG


def test_dump_keeps_catalog_format():
    text = fashion_links.CATALOG_PATH.read_text(encoding="utf-8")
    assert fashion_links.dump_catalog(json.loads(text)) + "\n" == text
