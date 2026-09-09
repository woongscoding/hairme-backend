"""
scripts/backfill_feedback_to_s3.py 회귀 테스트

검증 항목:
1. NPZ 레이아웃이 S3FeedbackStore.save_feedback 결과와 동일 (keys/dtypes/shapes)
2. 중복 판정(dedupe) 로직 - 구버전(스타일 접미사 없음) / 신규 키
3. DynamoDB Decimal -> float 변환
4. 라벨 매핑 (good/like -> 90.0, bad/dislike -> 10.0)
5. dry-run 은 S3에 아무것도 쓰지 않음
6. --execute 는 N개 객체를 쓰고 metadata.json 카운터를 실제 객체 수로 재계산
7. --execute 안전 가드 (버킷 미지정 시 거부)

AWS 는 전부 MagicMock/페이크로 대체하며 실제 호출은 하지 않는다.
"""

import importlib.util
import io
import json
import sys
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import services.mlops.s3_feedback_store as sfs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "backfill_feedback_to_s3.py"


def _load_backfill_module():
    spec = importlib.util.spec_from_file_location(
        "backfill_feedback_to_s3", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bf = _load_backfill_module()


# ========== 페이크 AWS ==========


class FakeS3Client:
    """list_objects_v2 페이지네이션까지 흉내내는 최소 S3 목"""

    def __init__(self, objects=None, metadata=None, page_size=2):
        self.objects = dict(objects or {})
        self.metadata_body = (
            json.dumps(metadata).encode("utf-8") if metadata is not None else None
        )
        self.page_size = page_size
        self.put_calls = []

    def list_objects_v2(self, Bucket=None, Prefix="", ContinuationToken=None, **kwargs):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start : start + self.page_size]
        next_start = start + len(page)
        truncated = next_start < len(keys)

        response = {"Contents": [{"Key": k} for k in page], "IsTruncated": truncated}
        if truncated:
            response["NextContinuationToken"] = str(next_start)
        return response

    def get_object(self, Bucket=None, Key=None, **kwargs):
        if Key == bf.S3_METADATA_KEY:
            if self.metadata_body is None:
                raise KeyError(Key)
            return {"Body": io.BytesIO(self.metadata_body)}
        if Key in self.objects:
            return {"Body": io.BytesIO(self.objects[Key])}
        raise KeyError(Key)

    def put_object(self, Bucket=None, Key=None, Body=None, ContentType=None, **kwargs):
        self.put_calls.append(Key)
        if Key == bf.S3_METADATA_KEY:
            self.metadata_body = Body if isinstance(Body, bytes) else Body.encode()
        else:
            self.objects[Key] = Body
        return {}

    def stored_metadata(self):
        return json.loads(self.metadata_body)


class FakeTable:
    """ExclusiveStartKey 페이지네이션을 흉내내는 DynamoDB Table 목"""

    def __init__(self, items, page_size=2):
        self.items = items
        self.page_size = page_size
        self.scan_kwargs = []

    def scan(self, **kwargs):
        self.scan_kwargs.append(kwargs)
        start = int((kwargs.get("ExclusiveStartKey") or {}).get("offset", 0))
        page = self.items[start : start + self.page_size]
        next_start = start + len(page)

        response = {"Items": page}
        if next_start < len(self.items):
            response["LastEvaluatedKey"] = {"offset": next_start}
        return response


class FakeStyleLookup:
    """hairstyle_id -> 384차원 임베딩"""

    def get_style_embedding(self, hairstyle_id):
        try:
            hid = int(hairstyle_id)
        except (TypeError, ValueError):
            return None
        if hid < 0 or hid > 199:
            return None
        return np.full(384, float(hid), dtype=np.float32)


def make_item(analysis_id, feedbacks, hairstyle_ids, with_mediapipe=True):
    """DynamoDB 행 (Decimal 포함) 생성"""
    item = {
        "analysis_id": analysis_id,
        "created_at": "2026-03-15T04:05:06+00:00",
        "face_shape": "계란형",
        "personal_color": "봄웜",
        "recommended_styles": [
            {"hairstyle_id": hid, "style_name": f"style_{i}"}
            for i, hid in enumerate(hairstyle_ids)
        ],
    }
    for k, value in feedbacks.items():
        item[f"style_{k}_feedback"] = value

    if with_mediapipe:
        item.update(
            {
                "mediapipe_face_ratio": Decimal("1.25"),
                "mediapipe_forehead_width": Decimal("110.5"),
                "mediapipe_cheekbone_width": Decimal("140.25"),
                "mediapipe_jaw_width": Decimal("115.75"),
                "mediapipe_forehead_ratio": Decimal("0.79"),
                "mediapipe_jaw_ratio": Decimal("0.82"),
                "mediapipe_ITA_value": Decimal("71.5"),
                "mediapipe_hue_value": Decimal("14.25"),
            }
        )
    return item


# ========== (1) NPZ 레이아웃 동일성 ==========


def _make_store(monkeypatch):
    """목 S3 를 사용하는 실제 S3FeedbackStore (store 자체 writer 확인용)"""
    monkeypatch.setenv("MLOPS_ENABLED", "true")

    fake_s3 = FakeS3Client(metadata={"total_feedback_count": 0, "pending_count": 0})
    fake_s3.head_bucket = lambda **kwargs: {}

    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = fake_s3
    monkeypatch.setattr(sfs, "boto3", mock_boto3)

    store = sfs.S3FeedbackStore(bucket_name="test-mlops-bucket")
    store.style_embeddings = np.arange(10 * 384, dtype=np.float32).reshape(10, 384)
    store.style_to_idx = {f"style_{i}": i for i in range(10)}
    assert store.enabled is True
    return store, fake_s3


def test_npz_layout_matches_store_writer(monkeypatch):
    """백필 NPZ 가 store 가 만든 NPZ 와 동일한 keys/dtypes/shapes 를 가져야 한다"""
    store, fake_s3 = _make_store(monkeypatch)

    face = [1.25, 110.5, 140.25, 115.75, 0.79, 0.82]
    skin = [71.5, 14.25]

    result = store.save_feedback(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=3,
        feedback="good",
        face_features=face,
        skin_features=skin,
    )
    assert result["success"] is True

    store_key = [k for k in fake_s3.objects if k.endswith(".npz")][0]
    store_npz = np.load(io.BytesIO(fake_s3.objects[store_key]), allow_pickle=False)

    from datetime import datetime, timezone

    backfill_bytes = bf.build_npz_bytes(
        analysis_id="abcdef1234567890",
        face_shape="계란형",
        skin_tone="봄웜",
        hairstyle_id=3,
        feedback="good",
        face_features=face,
        skin_features=skin,
        style_embedding=store.get_style_embedding(3),
        timestamp=datetime(2026, 3, 15, tzinfo=timezone.utc),
    )
    backfill_npz = np.load(io.BytesIO(backfill_bytes), allow_pickle=False)

    assert set(backfill_npz.files) == set(store_npz.files)

    for name in store_npz.files:
        assert backfill_npz[name].shape == store_npz[name].shape, name
        if name == "metadata":
            # 유니코드 문자열 길이(<U188 vs <U203)는 JSON 내용에 따라 달라진다
            assert backfill_npz[name].dtype.kind == store_npz[name].dtype.kind == "U"
        else:
            assert backfill_npz[name].dtype == store_npz[name].dtype, name
            np.testing.assert_array_equal(backfill_npz[name], store_npz[name], name)

    meta = json.loads(str(backfill_npz["metadata"][0]))
    assert meta["analysis_id"] == "abcdef1234567890"
    assert meta["hairstyle_id"] == 3
    assert meta["feedback"] == "good"
    assert meta["source"] == "backfill"


def test_backfill_npz_loads_without_pickle():
    """백필 NPZ 도 allow_pickle=False 로 로드 가능해야 한다 (trainer 호환)"""
    from datetime import datetime, timezone

    body = bf.build_npz_bytes(
        analysis_id="11112222-3333",
        face_shape="둥근형",
        skin_tone="여름쿨",
        hairstyle_id=5,
        feedback="bad",
        face_features=[1.2, 100.0, 130.0, 105.0, 0.78, 0.8],
        skin_features=[70.0, 15.0],
        style_embedding=np.zeros(384, dtype=np.float32),
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    data = np.load(io.BytesIO(body), allow_pickle=False)

    assert data["face_features"].shape == (6,)
    assert data["skin_features"].shape == (2,)
    assert data["style_embedding"].shape == (384,)
    assert float(data["ground_truth"][0]) == 10.0
    for name in data.files:
        assert data[name].dtype != np.dtype("O"), name


# ========== (2) 라벨 매핑 / Decimal 변환 ==========


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("good", "good"),
        ("like", "good"),
        ("LIKE", "good"),
        ("bad", "bad"),
        ("dislike", "bad"),
        ("", None),
        (None, None),
        ("maybe", None),
        (123, None),
    ],
)
def test_normalize_feedback(raw, expected):
    assert bf.normalize_feedback(raw) == expected


def test_ground_truth_mapping_matches_store():
    """store 와 동일하게 good=90.0 / bad=10.0"""
    assert bf.ground_truth_for("good") == 90.0
    assert bf.ground_truth_for("bad") == 10.0


def test_decimal_conversion_and_feature_order():
    """Decimal 이 float 로 변환되고 순서가 dynamodb_connection 과 동일해야 한다"""
    item = make_item("aaaa-1", {1: "good"}, [3])
    face, skin = bf.extract_features(item)

    assert face == [1.25, 110.5, 140.25, 115.75, 0.79, 0.82]
    assert skin == [71.5, 14.25]
    assert all(isinstance(v, float) for v in face + skin)


def test_extract_features_requires_all_fields():
    item = make_item("aaaa-1", {1: "good"}, [3])
    del item["mediapipe_hue_value"]
    assert bf.extract_features(item) is None

    item2 = make_item("aaaa-2", {1: "good"}, [3], with_mediapipe=False)
    assert bf.extract_features(item2) is None


def test_extract_hairstyle_id_skips_trending():
    item = make_item("aaaa-3", {1: "good"}, [None])
    assert bf.extract_hairstyle_id(item, 1) is None

    item2 = make_item("aaaa-4", {2: "good"}, [1, Decimal("42")])
    assert bf.extract_hairstyle_id(item2, 2) == 42
    assert bf.extract_hairstyle_id(item2, 5) is None


# ========== (3) 중복 판정 ==========


def test_dedupe_new_style_scheme():
    """신규 네이밍(_s{id}_)은 스타일 단위로 중복 판정"""
    keys = ["feedback/pending/2026-03-15_abcdef12_s7_a1b2c3.npz"]

    assert bf.is_already_present(keys, "abcdef12-1111-2222", 7) is True
    assert bf.is_already_present(keys, "abcdef12-1111-2222", 8) is False
    assert bf.is_already_present(keys, "99999999-1111-2222", 7) is False


def test_dedupe_legacy_key_blocks_whole_analysis():
    """스타일 접미사가 없는 구버전 키는 해당 analysis 전체를 스킵"""
    keys = ["feedback/processed/batch_1/2025-12-02_abcdef12.npz"]

    for style_id in (1, 2, 3):
        assert bf.is_already_present(keys, "abcdef12-1111-2222", style_id) is True


def test_dedupe_checks_processed_prefix_too():
    keys = ["feedback/processed/batch_9/2026-01-01_abcdef12_s4_zzzzzz.npz"]
    assert bf.is_already_present(keys, "abcdef12-1111", 4) is True
    assert bf.is_already_present(keys, "abcdef12-1111", 5) is False


def test_filename_uses_store_scheme_with_backfill_prefix():
    from datetime import datetime, timezone

    name = bf.build_filename(
        "abcdef12-1111-2222", 7, datetime(2026, 3, 15, tzinfo=timezone.utc)
    )

    assert name.startswith("backfill_2026-03-15_abcdef12_s7_")
    assert name.endswith(".npz")


# ========== (4) 페이지네이션 ==========


def test_list_keys_paginates():
    objects = {f"feedback/pending/f{i}.npz": b"x" for i in range(5)}
    fake_s3 = FakeS3Client(objects=objects, page_size=2)

    keys = bf.list_keys(fake_s3, "bucket", "feedback/pending/")
    assert len(keys) == 5


def test_scan_analyses_paginates_with_projection():
    items = [make_item(f"id-{i}", {1: "good"}, [1]) for i in range(5)]
    table = FakeTable(items, page_size=2)

    scanned = bf.scan_analyses(table)
    assert len(scanned) == 5

    first_kwargs = table.scan_kwargs[0]
    assert "ProjectionExpression" in first_kwargs
    names = first_kwargs["ExpressionAttributeNames"].values()
    assert "analysis_id" in names
    assert "style_3_feedback" in names
    assert "mediapipe_cheekbone_width" in names

    limited = bf.scan_analyses(FakeTable(items, page_size=2), limit=3)
    assert len(limited) == 3


# ========== (5)/(6)/(7) CLI 모드 ==========


def _cli_items():
    return [
        # style_1 은 이미 S3에 존재(_s7_), style_2 는 신규
        make_item("aaaaaaaa-1111", {1: "good", 2: "dislike"}, [7, 8]),
        # 신규
        make_item("cccccccc-2222", {1: "like"}, [3]),
        # mediapipe 누락 -> 스킵
        make_item("dddddddd-3333", {1: "good"}, [4], with_mediapipe=False),
        # 트렌드 스타일(hairstyle_id None) -> 스킵
        make_item("eeeeeeee-4444", {1: "good"}, [None]),
    ]


def _cli_fake_s3():
    return FakeS3Client(
        objects={
            "feedback/pending/2026-03-01_aaaaaaaa_s7_aaa111.npz": b"npz",
            "feedback/processed/batch_1/2026-02-01_ffffffff_s2_bbb222.npz": b"npz",
        },
        metadata={
            "total_feedback_count": 999,
            "pending_count": 999,
            "model_version": "v6",
            "last_training_at": "2026-02-02T00:00:00+00:00",
        },
        page_size=1,
    )


def _run_cli(argv, fake_s3, items):
    table = FakeTable(items, page_size=2)
    with patch.object(bf, "get_s3_client", return_value=fake_s3), patch.object(
        bf, "get_dynamodb_table", return_value=table
    ), patch.object(bf, "load_style_lookup", return_value=FakeStyleLookup()):
        return bf.main(argv)


def test_dry_run_writes_nothing():
    fake_s3 = _cli_fake_s3()
    before = dict(fake_s3.objects)

    exit_code = _run_cli(
        ["--dry-run", "--bucket", "test-bucket"], fake_s3, _cli_items()
    )

    assert exit_code == 0
    assert fake_s3.put_calls == []
    assert fake_s3.objects == before
    assert fake_s3.stored_metadata()["pending_count"] == 999


def test_execute_writes_objects_and_rewrites_metadata():
    fake_s3 = _cli_fake_s3()

    exit_code = _run_cli(
        ["--execute", "--bucket", "test-bucket"], fake_s3, _cli_items()
    )
    assert exit_code == 0

    written = [
        k
        for k in fake_s3.put_calls
        if k.startswith(bf.S3_PENDING_PREFIX) and k.endswith(".npz")
    ]
    # aaaaaaaa 의 style_2(id=8) + cccccccc 의 style_1(id=3) = 2건
    assert len(written) == 2
    assert all("backfill_" in k for k in written)

    # 내용 검증: 라벨 매핑 (dislike -> 10.0, like -> 90.0)
    labels = sorted(
        float(
            np.load(io.BytesIO(fake_s3.objects[k]), allow_pickle=False)["ground_truth"][
                0
            ]
        )
        for k in written
    )
    assert labels == [10.0, 90.0]

    # metadata 카운터가 실제 객체 수로 재계산되어야 한다
    metadata = fake_s3.stored_metadata()
    assert metadata["pending_count"] == 3  # 기존 1 + 신규 2
    assert metadata["total_feedback_count"] == 4  # pending 3 + processed 1
    # 다른 키는 보존
    assert metadata["model_version"] == "v6"
    assert metadata["last_training_at"] == "2026-02-02T00:00:00+00:00"
    # 재학습 트리거 관련 필드를 새로 만들지 않는다
    assert "training_triggered_at" not in metadata


def test_execute_requires_bucket(monkeypatch):
    """--execute 는 --bucket 또는 MLOPS_S3_BUCKET 없이는 거부되어야 한다"""
    monkeypatch.delenv("MLOPS_S3_BUCKET", raising=False)

    fake_s3 = _cli_fake_s3()
    with patch.object(bf, "get_s3_client", return_value=fake_s3), patch.object(
        bf, "get_dynamodb_table", return_value=FakeTable([], page_size=2)
    ), patch.object(bf, "load_style_lookup", return_value=FakeStyleLookup()):
        exit_code = bf.main(["--execute"])

    assert exit_code == 2
    assert fake_s3.put_calls == []


def test_execute_accepts_env_bucket(monkeypatch):
    monkeypatch.setenv("MLOPS_S3_BUCKET", "env-bucket")

    fake_s3 = _cli_fake_s3()
    exit_code = _run_cli(["--execute"], fake_s3, _cli_items())

    assert exit_code == 0
    assert any(k.startswith(bf.S3_PENDING_PREFIX) for k in fake_s3.put_calls)


def test_limit_option_restricts_rows():
    fake_s3 = _cli_fake_s3()
    items = _cli_items()

    with patch.object(bf, "get_s3_client", return_value=fake_s3), patch.object(
        bf, "get_dynamodb_table", return_value=FakeTable(items, page_size=2)
    ), patch.object(bf, "load_style_lookup", return_value=FakeStyleLookup()):
        assert bf.main(["--execute", "--bucket", "b", "--limit", "1"]) == 0

    written = [k for k in fake_s3.put_calls if k.endswith(".npz")]
    # 첫 행만 처리 -> style_1 은 중복, style_2 만 신규
    assert len(written) == 1


def test_build_samples_counts_skip_reasons():
    samples, stats = build = bf.build_samples(
        _cli_items(),
        ["feedback/pending/2026-03-01_aaaaaaaa_s7_aaa111.npz"],
        FakeStyleLookup(),
    )

    assert stats["rows_scanned"] == 4
    assert stats["rows_with_feedback"] == 4
    assert stats["skip_missing_mediapipe"] == 1
    assert stats["skip_no_hairstyle_id"] == 1
    assert stats["already_present"] == 1
    assert stats["would_write"] == 2
    assert stats["label_good"] == 1
    assert stats["label_bad"] == 1
    assert {s["month"] for s in samples} == {"2026-03"}
    assert build is not None


# ========== 임베딩 로더 (AWS 호출 없음) ==========


@pytest.mark.skipif(
    not (PROJECT_ROOT / "data_source" / "style_embeddings.npz").exists(),
    reason="스타일 임베딩 번들 없음",
)
def test_load_style_lookup_uses_local_embeddings(monkeypatch):
    monkeypatch.setattr(sfs, "LOCAL_EMBEDDINGS_PATH", "/nonexistent.npz")
    monkeypatch.setattr(sfs, "boto3", MagicMock(), raising=False)

    lookup = bf.load_style_lookup(
        str(PROJECT_ROOT / "data_source" / "style_embeddings.npz")
    )

    embedding = lookup.get_style_embedding(0)
    assert embedding is not None
    assert embedding.shape == (384,)
    assert lookup.s3_client is None
