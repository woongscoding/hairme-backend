"""미용실용 앱(Pro 계정) 손님 기록 테스트 (moto 가짜 DynamoDB·S3)

- Pro 계정만 (손님 계정 403 pro_only), 준비 안 됨 503 records_unavailable
- 손님 등록·수정 (입력 검증, 같은 디자이너 안에서 전화번호 중복 409)
- 목록 정렬, 보관 기간이 지난 기록 숨김
- 상담 결과 저장 (첫 결과에서 상담 생성, 상담 수, 30개 제한, 사진 저장 실패 시 기록 없음)
- 고른 스타일, 손님 삭제 (사진 먼저, 실패하면 기록 유지)
- 회원 탈퇴 때 손님 기록 전부 삭제
- 보관 기간 계산 (상담·손님 2년)
손님 계정(HairMe) 요청은 이 기능을 전혀 건드리지 않는지 함께 확인한다.
"""

import importlib.util
import io
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("GEMINI_API_KEY", "test_api_key_123456")
os.environ.setdefault("JWT_SECRET_KEY", "test_jwt_secret_key_for_tests_only")

import boto3
import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from moto import mock_aws
from PIL import Image

from config.settings import settings
from database import pro_customer_repository as repo_module
from database.pro_customer_repository import (
    ProCustomerRepository,
    VisitChangedError,
    customer_sk,
    expire_at_after,
    is_expired,
    phone_sk,
    visit_sk,
)
from main import app
from services import pro_records_service as service_module
from services.photo_storage_service import (
    PhotoStorageService,
    pro_record_photo_key,
    pro_records_prefix,
)
from services.pro_records_service import ProRecordsService
from tests.test_pro_account import PRO_USER, _customer_auth, _png_bytes, _pro_auth

pytestmark = pytest.mark.moto

REGION = "ap-northeast-2"
TABLE = "hairme-pro-customers"
BUCKET = "test-records-bucket"
DESIGNER = "pro-user-id"
OTHER_DESIGNER = "pro-user-2"
DAY = 86400


def _jpeg_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _create_table(ddb) -> None:
    ddb.create_table(
        TableName=TABLE,
        KeySchema=[
            {"AttributeName": "user_id", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "user_id", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


@pytest.fixture
def aws(monkeypatch):
    from api.endpoints import pro_customers

    monkeypatch.setenv("AWS_REGION", REGION)
    monkeypatch.setenv("PHOTO_S3_BUCKET", BUCKET)
    monkeypatch.setattr(settings, "PHOTO_S3_BUCKET", BUCKET)
    monkeypatch.setenv("DYNAMODB_PRO_CUSTOMERS_TABLE_NAME", TABLE)
    monkeypatch.setattr(pro_customers.limiter, "enabled", False)
    with mock_aws():
        ddb = boto3.resource("dynamodb", region_name=REGION)
        _create_table(ddb)
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(
            Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION}
        )
        service = ProRecordsService(ProCustomerRepository(), PhotoStorageService())
        with patch(
            "api.endpoints.pro_customers.get_pro_records_service",
            return_value=service,
        ):
            yield SimpleNamespace(
                table=ddb.Table(TABLE), s3=s3, service=service, ddb=ddb
            )


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def _items(env, user_id=DESIGNER):
    return env.table.query(
        KeyConditionExpression="user_id = :u",
        ExpressionAttributeValues={":u": user_id},
    )["Items"]


def _s3_keys(env, prefix="pro/records/"):
    response = env.s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def _create(client, name="김손님", phone="010-1234-5678", auth=None):
    return client.post(
        "/api/pro/customers",
        json={"name": name, "phone": phone},
        headers=auth or _pro_auth(),
    )


def _upload(
    client,
    customer_id,
    visit_id="visit-0001",
    style="레이어드 컷",
    gender="female",
    image=None,
    filename="a.jpg",
    content_type="image/jpeg",
    auth=None,
):
    return client.post(
        f"/api/pro/customers/{customer_id}/visits/{visit_id}/results",
        data={"style": style, "gender": gender},
        files={"image": (filename, image or _jpeg_bytes(), content_type)},
        headers=auth or _pro_auth(),
    )


# ========== Pro 계정만 ==========


ROUTES = [
    ("get", "/api/pro/customers", {}),
    ("post", "/api/pro/customers", {"json": {"name": "a", "phone": "01012345678"}}),
    ("patch", "/api/pro/customers/cust-00001", {"json": {"memo": "x"}}),
    ("delete", "/api/pro/customers/cust-00001", {}),
    ("get", "/api/pro/customers/cust-00001/visits", {}),
    (
        "post",
        "/api/pro/customers/cust-00001/visits/visit-0001/results",
        {
            "data": {"style": "a", "gender": "male"},
            "files": {"image": ("a.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        },
    ),
    ("put", "/api/pro/customers/cust-00001/visits/visit-0001/chosen", {"json": {}}),
]


class TestProOnly:
    @pytest.mark.parametrize("method,path,kwargs", ROUTES)
    def test_customer_account_is_403(self, client, method, path, kwargs):
        service = MagicMock()
        with patch(
            "api.endpoints.pro_customers.get_pro_records_service",
            return_value=service,
        ):
            response = getattr(client, method)(path, headers=_customer_auth(), **kwargs)
        assert response.status_code == 403
        body = response.json()
        assert body["error"] == "pro_only"
        assert body["message"]
        # 손님 계정 요청은 기록 저장소를 전혀 건드리지 않는다
        assert not [c for c in service.mock_calls if not c[0].startswith("__")]

    @pytest.mark.parametrize("method,path,kwargs", ROUTES)
    def test_requires_login(self, client, method, path, kwargs):
        assert getattr(client, method)(path, **kwargs).status_code == 401

    @pytest.mark.parametrize("method,path,kwargs", ROUTES)
    def test_table_not_configured_is_503(
        self, client, monkeypatch, method, path, kwargs
    ):
        monkeypatch.setenv("DYNAMODB_PRO_CUSTOMERS_TABLE_NAME", "")
        monkeypatch.setenv("PHOTO_S3_BUCKET", BUCKET)
        service = ProRecordsService(ProCustomerRepository(), PhotoStorageService())
        with patch(
            "api.endpoints.pro_customers.get_pro_records_service",
            return_value=service,
        ):
            response = getattr(client, method)(path, headers=_pro_auth(), **kwargs)
        assert response.status_code == 503
        assert response.json()["error"] == "records_unavailable"

    def test_bucket_not_configured_is_503(self, client, monkeypatch):
        monkeypatch.setenv("DYNAMODB_PRO_CUSTOMERS_TABLE_NAME", TABLE)
        monkeypatch.setenv("PHOTO_S3_BUCKET", "")
        service = ProRecordsService(ProCustomerRepository(), PhotoStorageService())
        with patch(
            "api.endpoints.pro_customers.get_pro_records_service",
            return_value=service,
        ):
            response = client.get("/api/pro/customers", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "records_unavailable"

    def test_table_not_created_yet_is_503(self, client, aws):
        aws.table.delete()
        response = client.get("/api/pro/customers", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "records_unavailable"


# ========== 손님 등록 ==========


class TestCreateCustomer:
    def test_creates_customer_and_phone_marker(self, client, aws):
        response = _create(client, name="  김손님 ", phone="010-1234-5678")
        assert response.status_code == 201, response.text
        body = response.json()
        assert set(body) == {
            "customer_id",
            "name",
            "phone",
            "memo",
            "created_at",
            "last_visit_at",
            "visit_count",
        }
        assert body["name"] == "김손님"
        assert body["phone"] == "01012345678"
        assert body["memo"] == ""
        assert body["last_visit_at"] is None
        assert body["visit_count"] == 0
        assert service_module.ID_PATTERN.fullmatch(body["customer_id"])
        created = datetime.fromisoformat(body["created_at"])
        assert created.tzinfo is not None

        items = {i["sk"]: i for i in _items(aws)}
        customer = items[customer_sk(body["customer_id"])]
        marker = items[phone_sk("01012345678")]
        assert marker["customer_id"] == body["customer_id"]
        # 상담이 없으면 등록일부터 2년
        expected = int((created + timedelta(days=730)).timestamp())
        assert abs(int(customer["expire_at"]) - expected) <= 1
        assert int(marker["expire_at"]) == int(customer["expire_at"])

    @pytest.mark.parametrize(
        "name,phone",
        [
            ("", "01012345678"),
            ("   ", "01012345678"),
            ("가" * 21, "01012345678"),
            ("줄\n바꿈", "01012345678"),
            (123, "01012345678"),
            (None, "01012345678"),
            ("김손님", "12345678901"),  # 0 으로 시작해야 한다
            ("김손님", "0101234"),  # 너무 짧다
            ("김손님", "010123456789"),  # 너무 길다
            ("김손님", "전화번호"),
            ("김손님", 1012345678),
            ("김손님", None),
        ],
    )
    def test_invalid_input_is_400(self, client, aws, name, phone):
        response = client.post(
            "/api/pro/customers",
            json={"name": name, "phone": phone},
            headers=_pro_auth(),
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_customer"
        assert _items(aws) == []

    def test_name_of_20_chars_is_ok(self, client, aws):
        assert _create(client, name="가" * 20).status_code == 201

    @pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b""])
    def test_malformed_body_is_400(self, client, aws, raw):
        response = client.post(
            "/api/pro/customers",
            content=raw,
            headers={**_pro_auth(), "Content-Type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_customer"

    def test_duplicate_phone_is_409_with_existing_customer(self, client, aws):
        first = _create(client, name="첫손님", phone="010-1111-2222").json()
        response = _create(client, name="다른이름", phone="01011112222")
        assert response.status_code == 409
        body = response.json()
        assert body["error"] == "duplicate_phone"
        assert body["message"]
        assert body["customer"] == first
        # 두 번째 손님은 만들어지지 않았다
        customers = [i for i in _items(aws) if i["sk"].startswith("C#")]
        assert len(customers) == 1

    def test_same_phone_for_other_designer_is_ok(self, client, aws):
        assert _create(client, phone="01011112222").status_code == 201
        response = _create(client, phone="01011112222", auth=_pro_auth(OTHER_DESIGNER))
        assert response.status_code == 201

    def test_phone_of_expired_customer_can_be_reused(self, client, aws):
        old = _create(client, phone="01011112222").json()
        past = int(datetime.now(timezone.utc).timestamp()) - 10
        for sk in (customer_sk(old["customer_id"]), phone_sk("01011112222")):
            aws.table.update_item(
                Key={"user_id": DESIGNER, "sk": sk},
                UpdateExpression="SET expire_at = :e",
                ExpressionAttributeValues={":e": past},
            )
        response = _create(client, name="새손님", phone="01011112222")
        assert response.status_code == 201
        marker = aws.table.get_item(
            Key={"user_id": DESIGNER, "sk": phone_sk("01011112222")}
        )["Item"]
        assert marker["customer_id"] == response.json()["customer_id"]

    def test_marker_without_customer_is_replaced(self, client, aws):
        # 지난 삭제가 중간에 끊겨 주인 없는 마커만 남은 경우
        future = int(datetime.now(timezone.utc).timestamp()) + 100 * DAY
        aws.table.put_item(
            Item={
                "user_id": DESIGNER,
                "sk": phone_sk("01011112222"),
                "customer_id": "gone-customer-1",
                "expire_at": future,
            }
        )
        response = _create(client, phone="01011112222")
        assert response.status_code == 201, response.text


# ========== 손님 목록 ==========


class TestListCustomers:
    def test_newest_activity_first_and_expired_hidden(self, client, aws):
        now = datetime.now(timezone.utc)
        future = int(now.timestamp()) + 100 * DAY

        def put(cid, created_days_ago, last_visit_days_ago=None, expire_at=future):
            item = {
                "user_id": DESIGNER,
                "sk": customer_sk(cid),
                "name": cid,
                "phone": "0101234567" + cid[-1],
                "memo": "",
                "created_at": (now - timedelta(days=created_days_ago)).isoformat(),
                "visit_count": 0,
                "expire_at": expire_at,
            }
            if last_visit_days_ago is not None:
                item["last_visit_at"] = (
                    now - timedelta(days=last_visit_days_ago)
                ).isoformat()
                item["visit_count"] = 1
            aws.table.put_item(Item=item)

        put("cust-aaaa1", created_days_ago=50, last_visit_days_ago=1)
        put("cust-aaaa2", created_days_ago=3)
        put("cust-aaaa3", created_days_ago=60, last_visit_days_ago=10)
        put("cust-aaaa4", created_days_ago=0.5)
        put("cust-aaaa5", created_days_ago=1, expire_at=int(now.timestamp()) - 1)
        # 다른 디자이너의 손님은 보이지 않는다
        aws.table.put_item(
            Item={
                "user_id": OTHER_DESIGNER,
                "sk": customer_sk("cust-other1"),
                "name": "x",
                "phone": "01099998888",
                "created_at": now.isoformat(),
                "visit_count": 0,
                "expire_at": future,
            }
        )

        response = client.get("/api/pro/customers", headers=_pro_auth())
        assert response.status_code == 200
        customers = response.json()["customers"]
        assert [c["customer_id"] for c in customers] == [
            "cust-aaaa4",
            "cust-aaaa1",
            "cust-aaaa2",
            "cust-aaaa3",
        ]
        assert customers[1]["visit_count"] == 1
        assert customers[0]["last_visit_at"] is None

    def test_empty(self, client, aws):
        response = client.get("/api/pro/customers", headers=_pro_auth())
        assert response.json() == {"customers": []}


# ========== 손님 수정 ==========


class TestUpdateCustomer:
    def test_updates_name_and_memo(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = client.patch(
            f"/api/pro/customers/{cid}",
            json={"name": " 새이름 ", "memo": "  두피 민감\n염색 주의 "},
            headers=_pro_auth(),
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["name"] == "새이름"
        assert body["memo"] == "두피 민감\n염색 주의"
        listed = client.get("/api/pro/customers", headers=_pro_auth()).json()
        assert listed["customers"][0] == body

    def test_phone_change_swaps_marker(self, client, aws):
        cid = _create(client, phone="01011112222").json()["customer_id"]
        response = client.patch(
            f"/api/pro/customers/{cid}",
            json={"phone": "010-3333-4444"},
            headers=_pro_auth(),
        )
        assert response.status_code == 200
        assert response.json()["phone"] == "01033334444"
        sks = {i["sk"] for i in _items(aws)}
        assert phone_sk("01033334444") in sks
        assert phone_sk("01011112222") not in sks
        # 예전 번호는 다른 손님이 쓸 수 있다
        assert _create(client, phone="01011112222").status_code == 201

    def test_same_phone_is_noop(self, client, aws):
        created = _create(client, phone="01011112222").json()
        response = client.patch(
            f"/api/pro/customers/{created['customer_id']}",
            json={"phone": "010-1111-2222"},
            headers=_pro_auth(),
        )
        assert response.status_code == 200
        assert response.json() == created

    def test_duplicate_phone_on_update_is_409(self, client, aws):
        first = _create(client, name="첫손님", phone="01011112222").json()
        second = _create(client, name="둘째", phone="01033334444").json()
        response = client.patch(
            f"/api/pro/customers/{second['customer_id']}",
            json={"phone": "01011112222", "name": "바뀌면안됨"},
            headers=_pro_auth(),
        )
        assert response.status_code == 409
        assert response.json()["error"] == "duplicate_phone"
        assert response.json()["customer"] == first
        item = aws.table.get_item(
            Key={"user_id": DESIGNER, "sk": customer_sk(second["customer_id"])}
        )["Item"]
        assert item["phone"] == "01033334444"
        assert item["name"] == "둘째"

    @pytest.mark.parametrize(
        "body",
        [
            {"memo": "가" * 501},
            {"memo": 3},
            {"name": ""},
            {"phone": "123"},
        ],
    )
    def test_invalid_is_400(self, client, aws, body):
        cid = _create(client).json()["customer_id"]
        response = client.patch(
            f"/api/pro/customers/{cid}", json=body, headers=_pro_auth()
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_customer"

    def test_memo_of_500_chars_is_ok(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = client.patch(
            f"/api/pro/customers/{cid}", json={"memo": "가" * 500}, headers=_pro_auth()
        )
        assert response.status_code == 200

    @pytest.mark.parametrize("cid", ["cust-unknown1", "bad", "a" * 41])
    def test_unknown_customer_is_404(self, client, aws, cid):
        response = client.patch(
            f"/api/pro/customers/{cid}", json={"memo": "x"}, headers=_pro_auth()
        )
        assert response.status_code == 404
        assert response.json()["error"] == "customer_not_found"

    def test_other_designers_customer_is_404(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = client.patch(
            f"/api/pro/customers/{cid}",
            json={"memo": "x"},
            headers=_pro_auth(OTHER_DESIGNER),
        )
        assert response.status_code == 404

    def test_expired_customer_is_404(self, client, aws):
        cid = _create(client).json()["customer_id"]
        aws.table.update_item(
            Key={"user_id": DESIGNER, "sk": customer_sk(cid)},
            UpdateExpression="SET expire_at = :e",
            ExpressionAttributeValues={":e": 1},
        )
        response = client.patch(
            f"/api/pro/customers/{cid}", json={"memo": "x"}, headers=_pro_auth()
        )
        assert response.status_code == 404


# ========== 상담 결과 저장 ==========


class TestAddResult:
    def test_first_result_creates_visit(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = _upload(client, cid, visit_id="visit-0001")
        assert response.status_code == 201, response.text
        visit = response.json()["visit"]
        assert visit["visit_id"] == "visit-0001"
        assert visit["customer_id"] == cid
        assert visit["gender"] == "female"
        assert visit["chosen_style"] is None
        assert len(visit["results"]) == 1
        result = visit["results"][0]
        assert set(result) == {"result_id", "style", "photo_url"}
        assert result["style"] == "레이어드 컷"
        assert result["photo_url"].startswith("https://")

        key = pro_record_photo_key(
            DESIGNER, cid, "visit-0001", result["result_id"], "jpg"
        )
        assert _s3_keys(aws) == [key]
        head = aws.s3.head_object(Bucket=BUCKET, Key=key)
        assert head["ContentType"] == "image/jpeg"

        items = {i["sk"]: i for i in _items(aws)}
        customer = items[customer_sk(cid)]
        assert int(customer["visit_count"]) == 1
        assert customer["last_visit_at"] == visit["at"]
        stored = items[visit_sk(cid, "visit-0001")]
        assert stored["results"][0]["photo_key"] == key
        # 상담은 상담일부터, 손님·마커는 마지막 상담일부터 2년
        at = datetime.fromisoformat(visit["at"])
        expected = int((at + timedelta(days=730)).timestamp())
        assert int(stored["expire_at"]) == expected
        assert int(customer["expire_at"]) == expected
        assert int(items[phone_sk("01012345678")]["expire_at"]) == expected

    def test_more_results_append_in_order_without_counting_visit(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid, style="A")
        _upload(client, cid, style="B", gender="male")
        response = _upload(client, cid, style="C")
        visit = response.json()["visit"]
        assert [r["style"] for r in visit["results"]] == ["A", "B", "C"]
        assert visit["gender"] == "female"  # 첫 결과의 성별을 유지
        customer = client.get("/api/pro/customers", headers=_pro_auth()).json()[
            "customers"
        ][0]
        assert customer["visit_count"] == 1

    def test_new_visit_counts(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid, visit_id="visit-0001")
        response = _upload(client, cid, visit_id="visit-0002")
        assert response.status_code == 201
        customer = client.get("/api/pro/customers", headers=_pro_auth()).json()[
            "customers"
        ][0]
        assert customer["visit_count"] == 2
        assert customer["last_visit_at"] == response.json()["visit"]["at"]

    def test_png_is_stored_as_png_by_magic_bytes(self, client, aws):
        cid = _create(client).json()["customer_id"]
        # 앱이 image/jpeg 라고 보내도 내용이 PNG 면 PNG 로 저장한다
        response = _upload(client, cid, image=_png_bytes(), content_type="image/jpeg")
        assert response.status_code == 201
        (key,) = _s3_keys(aws)
        assert key.endswith(".png")
        assert aws.s3.head_object(Bucket=BUCKET, Key=key)["ContentType"] == "image/png"

    @pytest.mark.parametrize(
        "visit_id",
        ["short", "a" * 41, "has space1", "has.dot123", "한글상담아이디입니다"],
    )
    def test_invalid_visit_id_is_400(self, client, aws, visit_id):
        cid = _create(client).json()["customer_id"]
        response = _upload(client, cid, visit_id=visit_id)
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_visit"
        assert _s3_keys(aws) == []

    def test_visit_id_with_allowed_chars(self, client, aws):
        cid = _create(client).json()["customer_id"]
        assert _upload(client, cid, visit_id="Ab_-09xyz").status_code == 201

    @pytest.mark.parametrize(
        "style,gender",
        [("", "female"), ("가" * 51, "female"), ("A", "other"), ("A", "")],
    )
    def test_invalid_style_or_gender_is_400(self, client, aws, style, gender):
        cid = _create(client).json()["customer_id"]
        response = _upload(client, cid, style=style, gender=gender)
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_result"
        assert _s3_keys(aws) == []

    @pytest.mark.parametrize(
        "image",
        [
            b"GIF89a....",
            b"RIFF\x00\x00\x00\x00WEBPVP8 ",  # webp 는 받지 않는다
            b"",
            b"\xff\xd8\xff" + b"0" * (3 * 1024 * 1024),  # 3MB 초과
        ],
        ids=["gif", "webp", "empty", "over_3mb"],
    )
    def test_invalid_image_is_400(self, client, aws, image):
        cid = _create(client).json()["customer_id"]
        response = client.post(
            f"/api/pro/customers/{cid}/visits/visit-0001/results",
            data={"style": "A", "gender": "male"},
            files={"image": ("a.jpg", image, "image/jpeg")},
            headers=_pro_auth(),
        )
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_image"
        assert _s3_keys(aws) == []

    def test_missing_image_is_400(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = client.post(
            f"/api/pro/customers/{cid}/visits/visit-0001/results",
            data={"style": "A", "gender": "male"},
            headers=_pro_auth(),
        )
        assert response.status_code == 400

    def test_unknown_customer_is_404(self, client, aws):
        response = _upload(client, "cust-unknown1")
        assert response.status_code == 404
        assert response.json()["error"] == "customer_not_found"
        assert _s3_keys(aws) == []

    def test_31st_result_is_409(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid)
        aws.table.update_item(
            Key={"user_id": DESIGNER, "sk": visit_sk(cid, "visit-0001")},
            UpdateExpression="SET results = :r",
            ExpressionAttributeValues={
                ":r": [
                    {"result_id": f"r{i}", "style": f"S{i}", "photo_key": "k"}
                    for i in range(30)
                ]
            },
        )
        before = _s3_keys(aws)
        response = _upload(client, cid)
        assert response.status_code == 409
        assert response.json()["error"] == "too_many_results"
        assert _s3_keys(aws) == before

    def test_photo_save_failure_writes_no_record(self, client, aws):
        cid = _create(client).json()["customer_id"]
        with patch.object(
            aws.service.storage,
            "save_pro_record_photo",
            side_effect=RuntimeError("s3 down"),
        ):
            response = _upload(client, cid)
        assert response.status_code == 503
        assert response.json()["error"] == "save_failed"
        sks = {i["sk"] for i in _items(aws)}
        assert visit_sk(cid, "visit-0001") not in sks
        customer = aws.table.get_item(
            Key={"user_id": DESIGNER, "sk": customer_sk(cid)}
        )["Item"]
        assert int(customer["visit_count"]) == 0
        assert "last_visit_at" not in customer

    def test_record_write_failure_removes_photo(self, client, aws):
        cid = _create(client).json()["customer_id"]
        error = ClientError(
            {"Error": {"Code": "InternalServerError", "Message": "x"}},
            "TransactWriteItems",
        )
        with patch.object(
            aws.service.repository, "add_visit_result", side_effect=error
        ):
            response = _upload(client, cid)
        assert response.status_code == 503
        assert response.json()["error"] == "save_failed"
        assert _s3_keys(aws) == []

    def test_concurrent_visit_creation_retries_as_append(self, aws):
        cid = aws.service.create_customer(DESIGNER, "a", "01012345678")["customer_id"]
        aws.service.add_result(
            DESIGNER, cid, "visit-0001", "A", "male", _jpeg_bytes(), "jpeg"
        )
        repo = aws.service.repository
        real_get_visit = repo.get_visit
        calls = []

        def stale_first(*args, **kwargs):
            calls.append(1)
            # 첫 읽기는 "아직 상담 없음" 이었다고 가정 (다른 요청이 막 만든 상태)
            return None if len(calls) == 1 else real_get_visit(*args, **kwargs)

        with patch.object(repo, "get_visit", side_effect=stale_first):
            visit = aws.service.add_result(
                DESIGNER, cid, "visit-0001", "B", "male", _jpeg_bytes(), "jpeg"
            )
        assert [r["style"] for r in visit["results"]] == ["A", "B"]
        customer = repo.get_customer(DESIGNER, cid, datetime.now(timezone.utc))
        assert customer["visit_count"] == 1

    def test_stale_result_count_is_rejected(self, aws):
        cid = aws.service.create_customer(DESIGNER, "a", "01012345678")["customer_id"]
        aws.service.add_result(
            DESIGNER, cid, "visit-0001", "A", "male", _jpeg_bytes(), "jpeg"
        )
        repo = aws.service.repository
        now = datetime.now(timezone.utc)
        customer = repo.get_customer(DESIGNER, cid, now)
        visit = repo.get_visit(DESIGNER, cid, "visit-0001", now)
        stale = {**visit, "results": []}
        with pytest.raises(VisitChangedError):
            repo.add_visit_result(
                DESIGNER,
                customer,
                "visit-0001",
                stale,
                {"result_id": "x", "style": "B", "photo_key": "k"},
                "male",
                now,
            )

    def test_result_on_old_visit_keeps_later_last_visit(self, aws):
        """오래된 상담에 결과를 더해도 마지막 상담일·보관 기간이 줄지 않는다"""
        cid = aws.service.create_customer(DESIGNER, "a", "01012345678")["customer_id"]
        repo = aws.service.repository
        now = datetime.now(timezone.utc)
        old_at = now - timedelta(days=100)
        aws.table.put_item(
            Item={
                "user_id": DESIGNER,
                "sk": visit_sk(cid, "visit-old01"),
                "at": old_at.isoformat(),
                "gender": "male",
                "results": [{"result_id": "r0", "style": "A", "photo_key": "k"}],
                "expire_at": expire_at_after(old_at),
            }
        )
        aws.service.add_result(
            DESIGNER, cid, "visit-new01", "B", "male", _jpeg_bytes(), "jpeg"
        )
        before = repo.get_customer(DESIGNER, cid, now)
        aws.service.add_result(
            DESIGNER, cid, "visit-old01", "C", "male", _jpeg_bytes(), "jpeg"
        )
        after = repo.get_customer(DESIGNER, cid, now)
        assert after["last_visit_at"] == before["last_visit_at"]
        assert after["expire_at"] == before["expire_at"]
        assert after["visit_count"] == 1


# ========== 상담 기록 목록 ==========


class TestListVisits:
    def test_newest_first_and_expired_hidden(self, client, aws):
        cid = _create(client).json()["customer_id"]
        now = datetime.now(timezone.utc)

        def put(visit_id, days_ago, expire_at=None):
            at = now - timedelta(days=days_ago)
            aws.table.put_item(
                Item={
                    "user_id": DESIGNER,
                    "sk": visit_sk(cid, visit_id),
                    "at": at.isoformat(),
                    "gender": "male",
                    "results": [
                        {"result_id": "r1", "style": "A", "photo_key": "pro/x.jpg"}
                    ],
                    "chosen_style": "A",
                    "expire_at": expire_at or expire_at_after(at),
                }
            )

        put("visit-mid01", 30)
        put("visit-new01", 1)
        put("visit-old01", 300)
        put("visit-exp01", 731)  # 상담일부터 2년이 지났다 (TTL 이 아직 안 지움)
        response = client.get(f"/api/pro/customers/{cid}/visits", headers=_pro_auth())
        assert response.status_code == 200
        visits = response.json()["visits"]
        assert [v["visit_id"] for v in visits] == [
            "visit-new01",
            "visit-mid01",
            "visit-old01",
        ]
        assert set(visits[0]) == {
            "visit_id",
            "customer_id",
            "at",
            "gender",
            "chosen_style",
            "results",
        }
        assert visits[0]["chosen_style"] == "A"
        assert visits[0]["results"][0]["photo_url"].startswith("https://")

    def test_presign_failure_is_null_url(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid)
        with patch.object(aws.service.storage, "presigned_url", return_value=None):
            response = client.get(
                f"/api/pro/customers/{cid}/visits", headers=_pro_auth()
            )
        assert response.json()["visits"][0]["results"][0]["photo_url"] is None

    def test_presigned_url_lifetime(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid)
        with patch.object(
            aws.service.storage, "presigned_url", return_value="u"
        ) as presign:
            client.get(f"/api/pro/customers/{cid}/visits", headers=_pro_auth())
        assert presign.call_args.args[1] == 3600

    def test_unknown_customer_is_404(self, client, aws):
        response = client.get(
            "/api/pro/customers/cust-unknown1/visits", headers=_pro_auth()
        )
        assert response.status_code == 404
        assert response.json()["error"] == "customer_not_found"

    def test_no_visits(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = client.get(f"/api/pro/customers/{cid}/visits", headers=_pro_auth())
        assert response.json() == {"visits": []}


# ========== 고른 스타일 ==========


class TestChosenStyle:
    def _put(self, client, cid, style, visit_id="visit-0001"):
        return client.put(
            f"/api/pro/customers/{cid}/visits/{visit_id}/chosen",
            json={"style": style},
            headers=_pro_auth(),
        )

    def test_sets_chosen_style(self, client, aws):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid, style="A")
        _upload(client, cid, style="B")
        response = self._put(client, cid, "B")
        assert response.status_code == 200, response.text
        assert response.json()["visit"]["chosen_style"] == "B"
        visits = client.get(
            f"/api/pro/customers/{cid}/visits", headers=_pro_auth()
        ).json()["visits"]
        assert visits[0]["chosen_style"] == "B"

    @pytest.mark.parametrize("style", ["C", "", None, 3])
    def test_style_not_in_results_is_400(self, client, aws, style):
        cid = _create(client).json()["customer_id"]
        _upload(client, cid, style="A")
        response = self._put(client, cid, style)
        assert response.status_code == 400
        assert response.json()["error"] == "invalid_style"

    def test_unknown_visit_is_404(self, client, aws):
        cid = _create(client).json()["customer_id"]
        response = self._put(client, cid, "A", visit_id="visit-none1")
        assert response.status_code == 404
        assert response.json()["error"] == "visit_not_found"

    def test_unknown_customer_is_404(self, client, aws):
        response = self._put(client, "cust-unknown1", "A")
        assert response.status_code == 404
        assert response.json()["error"] == "customer_not_found"


# ========== 손님 삭제 ==========


class TestDeleteCustomer:
    def _setup(self, client, aws):
        cid = _create(client, phone="01011112222").json()["customer_id"]
        _upload(client, cid, visit_id="visit-0001")
        _upload(client, cid, visit_id="visit-0002")
        other = _create(client, name="남는손님", phone="01033334444").json()
        _upload(client, other["customer_id"])
        return cid, other

    def test_deletes_photos_visits_marker_and_customer(self, client, aws):
        cid, other = self._setup(client, aws)
        response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert response.status_code == 200
        assert response.json() == {"success": True}

        assert _s3_keys(aws, pro_records_prefix(DESIGNER, cid)) == []
        assert len(_s3_keys(aws, pro_records_prefix(DESIGNER, other["customer_id"])))
        sks = {i["sk"] for i in _items(aws)}
        assert not [sk for sk in sks if cid in sk]
        assert phone_sk("01011112222") not in sks
        assert phone_sk("01033334444") in sks
        customers = client.get("/api/pro/customers", headers=_pro_auth()).json()
        assert [c["customer_id"] for c in customers["customers"]] == [
            other["customer_id"]
        ]
        # 번호를 다시 쓸 수 있다
        assert _create(client, phone="01011112222").status_code == 201

    def test_s3_failure_keeps_records(self, client, aws):
        cid, _ = self._setup(client, aws)
        before = _items(aws)
        with patch.object(
            aws.service.storage,
            "delete_pro_records",
            side_effect=RuntimeError("s3 down"),
        ):
            response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "delete_failed"
        assert _items(aws) == before

    def test_photos_deleted_before_records(self, client, aws):
        cid, _ = self._setup(client, aws)
        order = []
        storage, repo = aws.service.storage, aws.service.repository
        real_storage, real_repo = (
            storage.delete_pro_records,
            repo.delete_customer_items,
        )

        def s3_delete(prefix):
            order.append("s3")
            return real_storage(prefix)

        def ddb_delete(*args):
            order.append("ddb")
            return real_repo(*args)

        with patch.object(
            storage, "delete_pro_records", side_effect=s3_delete
        ), patch.object(repo, "delete_customer_items", side_effect=ddb_delete):
            client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert order == ["s3", "ddb"]

    def test_record_delete_failure_is_503(self, client, aws):
        cid, _ = self._setup(client, aws)
        with patch.object(
            aws.service.repository,
            "delete_customer_items",
            side_effect=RuntimeError("ddb down"),
        ):
            response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "delete_failed"

    @pytest.mark.parametrize("cid", ["cust-unknown1", "bad"])
    def test_unknown_customer_is_success(self, client, aws, cid):
        response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert response.status_code == 200
        assert response.json() == {"success": True}

    def test_twice_is_success(self, client, aws):
        cid, _ = self._setup(client, aws)
        for _ in range(2):
            response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
            assert response.status_code == 200

    def test_leftover_visits_of_missing_customer_are_cleaned(self, client, aws):
        cid, _ = self._setup(client, aws)
        aws.table.delete_item(Key={"user_id": DESIGNER, "sk": customer_sk(cid)})
        response = client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth())
        assert response.status_code == 200
        assert not [i for i in _items(aws) if cid in i["sk"]]

    def test_other_designer_cannot_delete(self, client, aws):
        cid, _ = self._setup(client, aws)
        client.delete(f"/api/pro/customers/{cid}", headers=_pro_auth(OTHER_DESIGNER))
        assert [i for i in _items(aws) if i["sk"] == customer_sk(cid)]
        assert _s3_keys(aws, pro_records_prefix(DESIGNER, cid))


# ========== 회원 탈퇴 ==========


@pytest.fixture
def withdraw_aws(aws):
    from api.endpoints import auth

    previous = auth.limiter.enabled
    auth.limiter.enabled = False
    user_repo = MagicMock()
    user_repo.get_by_id_consistent.return_value = dict(PRO_USER)
    results_storage = MagicMock()
    results_storage.delete_user_results.return_value = 0
    with patch("api.endpoints.auth.get_user_repository", return_value=user_repo), patch(
        "api.endpoints.auth.get_photo_storage_service", return_value=results_storage
    ), patch(
        "api.endpoints.auth.get_pro_records_service", return_value=aws.service
    ), patch(
        "api.endpoints.auth.log_structured"
    ) as log:
        yield SimpleNamespace(aws=aws, user_repo=user_repo, log=log)
    auth.limiter.enabled = previous


class TestWithdrawalDeletesRecords:
    def test_deletes_all_records_then_account(self, client, withdraw_aws):
        aws = withdraw_aws.aws
        for phone in ("01011112222", "01033334444"):
            cid = _create(client, phone=phone).json()["customer_id"]
            _upload(client, cid)
        other = _create(client, auth=_pro_auth(OTHER_DESIGNER)).json()
        _upload(client, other["customer_id"], auth=_pro_auth(OTHER_DESIGNER))

        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200, response.text
        assert response.json() == {
            "success": True,
            "deleted_results": 0,
            "deleted_records": 2,
        }
        assert _items(aws) == []
        assert _s3_keys(aws, pro_records_prefix(DESIGNER)) == []
        # 다른 디자이너의 기록은 그대로
        assert len(_items(aws, OTHER_DESIGNER)) == 3
        assert len(_s3_keys(aws, pro_records_prefix(OTHER_DESIGNER))) == 1
        withdraw_aws.user_repo.delete_account.assert_called_once()
        event = withdraw_aws.log.call_args.args[1]
        assert event["deleted_records"] == 2

    def test_record_photo_failure_keeps_account(self, client, withdraw_aws):
        aws = withdraw_aws.aws
        cid = _create(client).json()["customer_id"]
        _upload(client, cid)
        with patch.object(
            aws.service.storage,
            "delete_pro_records",
            side_effect=RuntimeError("s3 down"),
        ):
            response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 503
        assert response.json()["error"] == "withdrawal_failed"
        withdraw_aws.user_repo.delete_account.assert_not_called()
        assert len(_items(aws)) == 3

    def test_record_table_failure_keeps_account(self, client, withdraw_aws):
        aws = withdraw_aws.aws
        with patch.object(
            aws.service.repository, "delete_all", side_effect=RuntimeError("ddb")
        ):
            response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 503
        withdraw_aws.user_repo.delete_account.assert_not_called()

    def test_table_not_created_is_skipped(self, client, withdraw_aws):
        withdraw_aws.aws.table.delete()
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200
        assert response.json()["deleted_records"] == 0
        withdraw_aws.user_repo.delete_account.assert_called_once()

    def test_table_not_configured_is_skipped(self, client, withdraw_aws, monkeypatch):
        monkeypatch.setenv("DYNAMODB_PRO_CUSTOMERS_TABLE_NAME", "")
        with patch.object(
            withdraw_aws.aws.service.storage, "delete_pro_records"
        ) as s3_delete:
            response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200
        assert response.json()["deleted_records"] == 0
        s3_delete.assert_not_called()

    def test_already_withdrawn_still_cleans_records(self, client, withdraw_aws):
        aws = withdraw_aws.aws
        cid = _create(client).json()["customer_id"]
        _upload(client, cid)
        withdraw_aws.user_repo.get_by_id_consistent.return_value = None
        response = client.delete("/api/auth/me", headers=_pro_auth())
        assert response.status_code == 200
        assert _items(aws) == []
        assert _s3_keys(aws) == []

    def test_customer_account_withdrawal_unchanged(self, client, withdraw_aws):
        with patch.object(withdraw_aws.aws.service, "delete_all_records") as delete_all:
            response = client.delete("/api/auth/me", headers=_customer_auth())
        assert response.status_code == 403
        assert response.json()["error"] == "withdrawal_not_supported"
        delete_all.assert_not_called()


# ========== 보관 기간 ==========


class TestRetention:
    def test_two_years_in_epoch_seconds(self):
        moment = datetime(2026, 11, 11, 3, 0, 0, tzinfo=timezone.utc)
        assert expire_at_after(moment) == int(moment.timestamp()) + 730 * DAY
        assert settings.PRO_RECORD_RETENTION_DAYS == 730

    def test_expired_boundary(self):
        now = datetime(2026, 11, 11, tzinfo=timezone.utc)
        ts = int(now.timestamp())
        assert is_expired({"expire_at": ts}, now)
        assert is_expired({"expire_at": ts - 1}, now)
        assert not is_expired({"expire_at": ts + 1}, now)
        assert is_expired(None, now)

    def test_visit_bumps_customer_retention(self, aws):
        repo = aws.service.repository
        created_at = datetime.now(timezone.utc) - timedelta(days=400)
        with patch.object(service_module, "now_utc", return_value=created_at):
            cid = aws.service.create_customer(DESIGNER, "a", "01012345678")[
                "customer_id"
            ]
        now = datetime.now(timezone.utc)
        customer = repo.get_customer(DESIGNER, cid, now)
        assert customer["expire_at"] == expire_at_after(created_at)

        aws.service.add_result(
            DESIGNER, cid, "visit-0001", "A", "male", _jpeg_bytes(), "jpeg"
        )
        customer = repo.get_customer(DESIGNER, cid, now)
        visit_at = datetime.fromisoformat(customer["last_visit_at"])
        assert customer["expire_at"] == expire_at_after(visit_at)
        assert customer["expire_at"] > expire_at_after(created_at)
        marker = repo._get(DESIGNER, phone_sk("01012345678"))
        assert marker["expire_at"] == customer["expire_at"]

    def test_customer_hidden_after_two_years_without_visit(self, aws):
        created_at = datetime.now(timezone.utc) - timedelta(days=730, seconds=5)
        with patch.object(service_module, "now_utc", return_value=created_at):
            aws.service.create_customer(DESIGNER, "a", "01012345678")
        assert aws.service.list_customers(DESIGNER) == []

    def test_timestamps_are_utc_iso(self):
        moment = datetime(2026, 11, 11, 3, 0, 0, 5000, tzinfo=timezone.utc)
        assert repo_module.to_iso(moment) == "2026-11-11T03:00:00.005+00:00"


# ========== 사진 저장소 도우미 ==========


class TestRecordStorage:
    @pytest.mark.parametrize(
        "prefix",
        [
            "pro/records/",
            "pro/records//",
            "pro/results/u/",
            "results/u/",
            "pro/records/u",
        ],
    )
    def test_refuses_prefix_outside_one_designer(self, monkeypatch, prefix):
        monkeypatch.setenv("PHOTO_S3_BUCKET", BUCKET)
        storage = PhotoStorageService()
        storage._client = MagicMock()
        with pytest.raises(ValueError):
            storage.delete_pro_records(prefix)
        storage._client.list_objects_v2.assert_not_called()

    def test_prefix_does_not_match_longer_ids(self):
        assert pro_records_prefix("u1", "cust-0001") == "pro/records/u1/cust-0001/"
        assert pro_records_prefix("u1") == "pro/records/u1/"
        assert not pro_record_photo_key("u1", "cust-00012", "v", "r", "jpg").startswith(
            pro_records_prefix("u1", "cust-0001")
        )

    def test_records_prefix_is_outside_pro_results_lifecycle(self):
        lifecycle = _load_script("apply_pro_results_lifecycle")
        prefixes = {rule["Filter"]["Prefix"]: rule for rule in lifecycle.RULES}
        assert prefixes["pro/results/"]["Expiration"]["Days"] == 7
        assert prefixes["pro/records/"]["Expiration"]["Days"] == 731
        assert not pro_records_prefix("u").startswith("pro/results/")
        # 사진은 기록(2년)보다 먼저 지워지지 않는다
        assert lifecycle.RECORDS_DAYS > settings.PRO_RECORD_RETENTION_DAYS


# ========== 운영 스크립트 ==========


def _load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestScripts:
    def test_lifecycle_keeps_existing_rules_and_updates_ours(self):
        lifecycle = _load_script("apply_pro_results_lifecycle")
        existing = [
            {"ID": "expire-cache", "Filter": {"Prefix": "cache/"}},
            {"ID": "expire-pro-results", "Filter": {"Prefix": "old/"}},
        ]
        rules = lifecycle.merged_rules(existing)
        assert [r["ID"] for r in rules] == [
            "expire-cache",
            "expire-pro-results",
            "expire-pro-records",
        ]
        assert rules[1]["Filter"]["Prefix"] == "pro/results/"
        # 다시 돌려도 같은 결과
        assert lifecycle.merged_rules(rules) == rules

    def test_lifecycle_preview_does_not_write(self, monkeypatch, capsys):
        lifecycle = _load_script("apply_pro_results_lifecycle")
        s3 = MagicMock()
        s3.get_bucket_lifecycle_configuration.return_value = {"Rules": []}
        monkeypatch.setattr(sys, "argv", ["x", "--bucket", BUCKET])
        with patch("boto3.client", return_value=s3):
            assert lifecycle.main() == 0
        s3.put_bucket_lifecycle_configuration.assert_not_called()
        assert "pro/records/" in capsys.readouterr().out

    def test_table_script_preview_does_not_touch_aws(self, monkeypatch, capsys):
        script = _load_script("create_pro_customers_table")
        monkeypatch.setattr(sys, "argv", ["x"])
        with patch("boto3.client") as client_factory:
            assert script.main() == 0
        client_factory.assert_not_called()
        assert "미리보기" in capsys.readouterr().out

    def test_table_script_creates_table_with_ttl(self, monkeypatch):
        script = _load_script("create_pro_customers_table")
        monkeypatch.setattr(sys, "argv", ["x", "--execute", "--table", "t-records"])
        with mock_aws():
            assert script.main() == 0
            # 다시 돌려도 실패하지 않는다
            assert script.main() == 0
            client = boto3.client("dynamodb", region_name=REGION)
            table = client.describe_table(TableName="t-records")["Table"]
            assert table["BillingModeSummary"]["BillingMode"] == "PAY_PER_REQUEST"
            assert {k["AttributeName"]: k["KeyType"] for k in table["KeySchema"]} == {
                "user_id": "HASH",
                "sk": "RANGE",
            }
            ttl = client.describe_time_to_live(TableName="t-records")
            assert ttl["TimeToLiveDescription"]["AttributeName"] == "expire_at"
            assert ttl["TimeToLiveDescription"]["TimeToLiveStatus"] == "ENABLED"


# ========== 손님(HairMe) 경로 ==========


class TestCustomerAppUnchanged:
    def test_customer_withdrawal_and_profile_do_not_touch_records(self, client):
        service = MagicMock()
        repo = MagicMock()
        repo.get_by_id.return_value = {
            "user_id": "customer-user-id",
            "nickname": "손님",
            "credits": 3,
        }
        with patch(
            "api.endpoints.auth.get_pro_records_service", return_value=service
        ), patch("api.endpoints.auth.get_user_repository", return_value=repo):
            me = client.get("/api/auth/me", headers=_customer_auth())
            assert me.status_code == 200
            assert set(me.json()["user"]) == {
                "user_id",
                "nickname",
                "email",
                "credits",
                "paid_credits",
                "ad_free",
                "training_consent",
                "created_at",
            }
        service.delete_all_records.assert_not_called()

    def test_routes_are_registered_on_main_app_only(self):
        import main_beauty

        main_paths = {getattr(r, "path", "") for r in app.routes}
        assert "/api/pro/customers" in main_paths
        assert "/api/pro/customers/{customer_id}/visits/{visit_id}/chosen" in main_paths
        beauty_paths = {getattr(r, "path", "") for r in main_beauty.app.routes}
        assert not [p for p in beauty_paths if p.startswith("/api/pro/")]

    def test_json_error_bodies_have_message(self):
        from api.endpoints import pro_customers

        for response in (
            pro_customers._pro_only(),
            pro_customers._unavailable(),
            pro_customers._invalid_customer(),
            pro_customers._customer_not_found(),
            pro_customers._visit_not_found(),
            pro_customers._invalid_image(),
        ):
            body = json.loads(response.body)
            assert set(body) == {"error", "message"}
            assert body["message"]
