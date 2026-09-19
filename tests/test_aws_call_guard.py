# -*- coding: utf-8 -*-
"""AWS 호출 차단 장치 자체에 대한 테스트 (tests/conftest.py)

이 장치가 조용히 고장나면 목을 빠뜨린 테스트가 다시 운영 테이블에 붙는다.
대부분의 저장소 접근이 fail-open 으로 감싸여 있어 그래도 테스트는 통과하므로,
차단이 실제로 걸리는지 여기서 직접 확인한다.
"""

import os

import pytest

from tests.conftest import AWS_INTEGRATION_OPT_IN_ENV, RealAWSCallAttempted

# 옵트인 실행에서는 차단 장치가 의도적으로 꺼져 있다. 이 스위트를 그대로 돌리면
# "차단됐는지" 확인하려다 실제 AWS 를 호출하게 되므로 통째로 skip 한다.
pytestmark = pytest.mark.skipif(
    os.getenv(AWS_INTEGRATION_OPT_IN_ENV) == "1",
    reason=f"{AWS_INTEGRATION_OPT_IN_ENV}=1 이면 차단 장치가 꺼져 있어 검증 대상이 아님",
)


class TestGuardBlocksRealCalls:
    def test_dynamodb_call_is_blocked(self):
        import boto3

        table = boto3.resource("dynamodb", region_name="ap-northeast-2").Table(
            "hairstyle_usage"
        )

        with pytest.raises(RealAWSCallAttempted) as exc_info:
            table.get_item(Key={"device_id": "guard-test", "date": "2026-09-19"})

        assert "dynamodb" in str(exc_info.value)
        assert "GetItem" in str(exc_info.value)

    def test_other_aws_services_are_blocked_too(self):
        import boto3

        s3 = boto3.client("s3", region_name="ap-northeast-2")

        with pytest.raises(RealAWSCallAttempted):
            s3.list_buckets()

    def test_guard_survives_fail_open_handlers(self):
        """`except Exception` 으로 감싼 fail-open 경로에 삼켜지지 않아야 한다.

        core.synthesis_budget 처럼 저장소 장애를 통과시키는 코드가 많아,
        일반 Exception 이면 차단이 무력화된다.
        """
        from core.synthesis_budget import record_api_calls

        with pytest.raises(RealAWSCallAttempted):
            # 내부의 `except Exception` 이 잡지 못하고 그대로 올라와야 한다
            record_api_calls(1)

    def test_blocked_error_is_not_an_exception_subclass(self):
        assert issubclass(RealAWSCallAttempted, BaseException)
        assert not issubclass(RealAWSCallAttempted, Exception)


class TestGuardOptOut:
    @pytest.mark.aws
    def test_aws_marker_lifts_the_guard(self):
        """@pytest.mark.aws 는 차단을 풀어준다 (실제 호출은 하지 않고 확인만)"""
        import botocore.client

        # 차단이 걸려 있으면 _make_api_call 이 conftest 의 함수로 교체돼 있다
        assert botocore.client.BaseClient._make_api_call.__name__ != "_blocked"

    def test_guard_is_active_without_the_marker(self):
        import botocore.client

        assert botocore.client.BaseClient._make_api_call.__name__ == "_blocked"

    def test_integration_env_is_the_documented_opt_in(self):
        """통합 테스트 스위트와 같은 플래그를 쓴다"""
        from tests import test_dynamodb_integration as integration

        assert AWS_INTEGRATION_OPT_IN_ENV == integration.RUN_INTEGRATION_ENV
        # 기본 실행에서는 꺼져 있어야 한다
        assert os.getenv(AWS_INTEGRATION_OPT_IN_ENV) != "1"
