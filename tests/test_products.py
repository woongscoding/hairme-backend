"""제휴 제품 추천/클릭 테스트 (카탈로그 + 엔드포인트 + 클릭 로그)"""

import os

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from core.jwt_auth import create_access_token
from main import app
from services.affiliate_click_service import AffiliateClickService
from services.product_recommendation_service import (
    ProductRecommendationService,
    get_product_recommendation_service,
)


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers():
    token = create_access_token("click-user-id")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_click_service():
    service = MagicMock()
    service.log_click.return_value = True
    with patch(
        "api.endpoints.products.get_affiliate_click_service", return_value=service
    ):
        yield service


@pytest.fixture
def known_product_id():
    """카탈로그의 첫 활성 제품 ID (시드 데이터가 바뀌어도 테스트 유지)"""
    catalog = get_product_recommendation_service().catalog
    return catalog["products"][0]["product_id"]


class TestProductRecommendationService:
    def setup_method(self):
        self.service = ProductRecommendationService()

    def test_perm_style_recommends_curl_products_first(self):
        products = self.service.get_recommendations("C컬펌", gender="female")
        assert products
        assert products[0]["category"] == "curl_cream"

    def test_explicit_mapping_wins_over_keyword(self):
        # "포마드"는 style_mappings에 명시 - 키워드 규칙보다 우선
        products = self.service.get_recommendations("포마드", gender="male")
        assert products[0]["category"] == "pomade"

    def test_male_cut_gets_styling_products(self):
        products = self.service.get_recommendations("크롭컷", gender="male")
        assert products[0]["category"] in ("wax", "pomade")

    def test_female_cut_gets_care_products_not_wax(self):
        products = self.service.get_recommendations("레이어드컷", gender="female")
        assert products
        assert products[0]["category"] == "hair_oil"
        assert all(p["category"] not in ("wax", "pomade") for p in products)

    def test_male_perm_includes_styling_wax(self):
        products = self.service.get_recommendations("가르마펌", gender="male")
        assert products[0]["category"] == "curl_cream"
        assert any(p["category"] == "wax" for p in products)

    def test_down_perm_recommends_self_perm_kit_first(self):
        products = self.service.get_recommendations("다운펌", gender="male")
        assert products[0]["category"] == "down_perm_kit"

    def test_long_hair_style_recommends_oil_first(self):
        products = self.service.get_recommendations("장발스타일", gender="male")
        assert products[0]["category"] == "hair_oil"

    def test_unknown_style_falls_back_to_default(self):
        products = self.service.get_recommendations("존재하지않는스타일")
        assert products  # 빈 목록이 아니라 기본 케어 제품

    def test_no_style_returns_default(self):
        assert self.service.get_recommendations(None)

    def test_recommendations_exclude_affiliate_url(self):
        for product in self.service.get_recommendations("가르마펌", gender="male"):
            assert "affiliate_url" not in product

    def test_limit_respected(self):
        assert len(self.service.get_recommendations("C컬펌", limit=2)) <= 2

    def test_get_product_includes_affiliate_url(self):
        product_id = self.service.catalog["products"][0]["product_id"]
        product = self.service.get_product(product_id)
        assert product is not None
        assert product["affiliate_url"]

    def test_get_product_unknown_returns_none(self):
        assert self.service.get_product("no-such-product") is None

    def test_disclosure_present(self):
        assert "쿠팡 파트너스" in self.service.disclosure


class TestRecommendationsEndpoint:
    def test_returns_products_and_disclosure(self, client):
        response = client.get(
            "/api/products/recommendations", params={"style": "C컬펌"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["style"] == "C컬펌"
        assert len(data["products"]) <= 3
        assert data["products"]
        assert "쿠팡 파트너스" in data["disclosure"]
        for product in data["products"]:
            assert "affiliate_url" not in product

    def test_no_auth_required(self, client):
        response = client.get("/api/products/recommendations")
        assert response.status_code == 200
        assert response.json()["products"]

    def test_invalid_gender_rejected(self, client):
        response = client.get(
            "/api/products/recommendations", params={"gender": "unknown"}
        )
        assert response.status_code == 422


class TestClickEndpoint:
    def test_anonymous_gets_url_and_is_logged_by_device(
        self, client, mock_click_service, known_product_id
    ):
        """비로그인도 링크를 받고, device_id로 클릭 로그가 남는다.

        로그인 필수였던 동안 합성 성공의 84%(비로그인)가 링크를 못 받았다.
        """
        response = client.post(
            "/api/products/click",
            json={
                "product_id": known_product_id,
                "source": "synthesis_result",
                "device_id": "device-abc-12345678",
            },
        )

        assert response.status_code == 200
        assert response.json()["affiliate_url"].startswith("https://")

        args = mock_click_service.log_click.call_args.args
        assert args[0] == "device#device-abc-12345678"
        assert args[5] is False  # authenticated

    def test_anonymous_without_device_id_still_gets_url(
        self, client, mock_click_service, known_product_id
    ):
        """식별자가 없으면 로그만 생략하고 링크는 발급한다 (수익 기회 > 로그 1건)."""
        response = client.post(
            "/api/products/click", json={"product_id": known_product_id}
        )

        assert response.status_code == 200
        assert response.json()["affiliate_url"].startswith("https://")
        mock_click_service.log_click.assert_not_called()

    def test_malformed_device_id_skips_log_but_returns_url(
        self, client, mock_click_service, known_product_id
    ):
        response = client.post(
            "/api/products/click",
            json={"product_id": known_product_id, "device_id": "a"},
        )

        assert response.status_code == 200
        assert response.json()["affiliate_url"].startswith("https://")
        mock_click_service.log_click.assert_not_called()

    def test_token_wins_over_device_id(
        self, client, auth_headers, mock_click_service, known_product_id
    ):
        """토큰이 있으면 device_id가 같이 와도 user_id로 기록한다."""
        response = client.post(
            "/api/products/click",
            json={
                "product_id": known_product_id,
                "device_id": "device-abc-12345678",
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        args = mock_click_service.log_click.call_args.args
        assert args[0] == "click-user-id"
        assert args[5] is True  # authenticated

    def test_returns_affiliate_url_and_logs(
        self, client, auth_headers, mock_click_service, known_product_id
    ):
        body = {
            "product_id": known_product_id,
            "style": "C컬펌",
            "source": "synthesis_result",
            "hair_profile": {"face_shape": "계란형", "personal_color": "봄웜"},
        }
        response = client.post("/api/products/click", json=body, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["affiliate_url"].startswith("https://")
        assert "쿠팡 파트너스" in data["disclosure"]

        args = mock_click_service.log_click.call_args.args
        assert args[0] == "click-user-id"
        assert args[1]["product_id"] == known_product_id
        assert args[2] == "C컬펌"
        assert args[3] == "synthesis_result"
        assert args[4] == {"face_shape": "계란형", "personal_color": "봄웜"}
        assert args[5] is True  # authenticated

    def test_unknown_product_404(self, client, auth_headers, mock_click_service):
        response = client.post(
            "/api/products/click",
            json={"product_id": "no-such-product"},
            headers=auth_headers,
        )
        assert response.status_code == 404
        mock_click_service.log_click.assert_not_called()

    def test_invalid_source_rejected(
        self, client, auth_headers, mock_click_service, known_product_id
    ):
        response = client.post(
            "/api/products/click",
            json={"product_id": known_product_id, "source": "hacked"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    def test_log_failure_still_returns_url(
        self, client, auth_headers, mock_click_service, known_product_id
    ):
        mock_click_service.log_click.side_effect = RuntimeError("dynamo down")
        response = client.post(
            "/api/products/click",
            json={"product_id": known_product_id},
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert response.json()["affiliate_url"]


class TestAffiliateClickService:
    def _service_with_mock_table(self):
        service = AffiliateClickService()
        service._table = MagicMock()
        return service

    def test_log_click_writes_item_with_profile(self):
        service = self._service_with_mock_table()
        ok = service.log_click(
            "user-1",
            {"product_id": "curl-001", "category": "curl_cream", "price_krw": 9900},
            style="C컬펌",
            source="synthesis_result",
            hair_profile={"face_shape": "계란형", "confidence": 0.92},
        )

        assert ok is True
        item = service._table.put_item.call_args.kwargs["Item"]
        assert item["user_id"] == "user-1"
        assert item["product_id"] == "curl-001"
        assert item["style"] == "C컬펌"
        assert item["price_krw"] == 9900
        # DynamoDB는 float 불가 - Decimal 변환 확인
        assert isinstance(item["hair_profile"]["confidence"], Decimal)
        assert item["sk"].startswith(item["created_at"])

    def test_log_click_swallows_dynamo_errors(self):
        service = self._service_with_mock_table()
        service._table.put_item.side_effect = RuntimeError("boom")
        ok = service.log_click(
            "user-1", {"product_id": "curl-001"}, style=None, source="browse"
        )
        assert ok is False

    def test_oversized_hair_profile_dropped(self):
        service = self._service_with_mock_table()
        service.log_click(
            "user-1",
            {"product_id": "curl-001"},
            style=None,
            source="browse",
            hair_profile={"blob": "x" * 5000},
        )
        item = service._table.put_item.call_args.kwargs["Item"]
        assert "hair_profile" not in item


class TestSynthesisRecommendationHelper:
    def test_returns_products_for_style(self):
        from api.endpoints.synthesis import _safe_product_recommendations

        products = _safe_product_recommendations("가르마펌", "male")
        assert products
        assert all("affiliate_url" not in p for p in products)

    def test_never_raises(self):
        from api.endpoints.synthesis import _safe_product_recommendations

        with patch(
            "api.endpoints.synthesis.get_product_recommendation_service",
            side_effect=RuntimeError("catalog broken"),
        ):
            assert _safe_product_recommendations("C컬펌") == []
