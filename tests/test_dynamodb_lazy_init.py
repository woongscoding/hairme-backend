"""
DynamoDB 지연 재초기화 테스트

콜드스타트 때 init_dynamodb() 가 일시 장애로 실패해도 컨테이너 수명 내내
저장이 꺼지지 않고, 이후 호출에서 (최소 간격을 두고) 다시 초기화되어야 한다.
"""

from unittest.mock import MagicMock, patch

import pytest

import database.dynamodb_connection as ddb


@pytest.fixture
def fresh_ddb(monkeypatch):
    """모듈 전역 상태를 초기화하고 테스트 후 원래 값으로 되돌린다"""
    monkeypatch.setenv("USE_DYNAMODB", "true")
    monkeypatch.setattr(ddb, "dynamodb_resource", None)
    monkeypatch.setattr(ddb, "dynamodb_table", None)
    monkeypatch.setattr(ddb, "dynamodb_enabled", False)
    monkeypatch.setattr(ddb, "_last_init_attempt_at", None)
    monkeypatch.setattr(ddb, "redis_client", None)
    yield ddb


def _resource_with(table):
    resource = MagicMock()
    resource.Table.return_value = table
    return resource


def test_init_does_not_call_describe_table(fresh_ddb):
    table = MagicMock()
    with patch.object(ddb.boto3, "resource", return_value=_resource_with(table)):
        assert ddb.init_dynamodb() is True

    table.load.assert_not_called()
    assert ddb.dynamodb_enabled is True
    assert ddb.dynamodb_table is table


def test_init_failure_then_retry_lets_save_analysis_work(fresh_ddb, monkeypatch):
    table = MagicMock()
    clock = {"now": 1000.0}
    monkeypatch.setattr(ddb.time, "monotonic", lambda: clock["now"])

    resource_factory = MagicMock(
        side_effect=[
            RuntimeError("transient cold-start failure"),
            _resource_with(table),
        ]
    )
    with patch.object(ddb.boto3, "resource", resource_factory):
        # 1) 콜드스타트 초기화 실패
        assert ddb.init_dynamodb() is False
        assert ddb.dynamodb_enabled is False

        # 2) 재시도 간격 이내에는 다시 시도하지 않는다 (호출마다 재시도 금지)
        clock["now"] += 1
        assert ddb.save_analysis({"face_shape": "계란형"}) is None
        assert resource_factory.call_count == 1
        table.put_item.assert_not_called()

        # 3) 간격이 지나면 지연 재초기화 후 저장 성공
        clock["now"] += ddb._INIT_RETRY_INTERVAL_SECONDS
        analysis_id = ddb.save_analysis(
            {"face_shape": "계란형", "personal_color": "봄웜", "processing_time": 1.5}
        )

    assert analysis_id is not None
    assert resource_factory.call_count == 2
    assert ddb.dynamodb_enabled is True
    table.put_item.assert_called_once()
    item = table.put_item.call_args.kwargs["Item"]
    assert item["analysis_id"] == analysis_id
    assert item["face_shape"] == "계란형"


def test_never_initialized_lazily_initializes_on_first_call(fresh_ddb):
    """init_dynamodb() 가 한 번도 호출되지 않았어도 첫 호출에서 초기화된다"""
    table = MagicMock()
    table.get_item.return_value = {
        "Item": {"analysis_id": "a1", "face_shape": "둥근형"}
    }
    with patch.object(ddb.boto3, "resource", return_value=_resource_with(table)):
        result = ddb.get_analysis("a1")

    assert result == {"analysis_id": "a1", "face_shape": "둥근형"}


def test_disabled_flag_still_disables(fresh_ddb, monkeypatch):
    """USE_DYNAMODB 게이트는 그대로 유지된다"""
    monkeypatch.setenv("USE_DYNAMODB", "false")
    with patch.object(ddb.boto3, "resource") as resource_factory:
        assert ddb.save_feedback("a1", 1, "good", False) is False
    resource_factory.assert_not_called()
