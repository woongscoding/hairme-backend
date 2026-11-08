"""
get_secret 캐시 동작 테스트

일시 장애(스로틀/네트워크)로 돌아온 None 이 캐시되어 컨테이너 수명 내내
시크릿 조회가 막히면 안 된다. 성공한 값만 캐시한다.
"""

from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from config.secrets import get_secret


def setup_function():
    get_secret.cache_clear()


def teardown_function():
    get_secret.cache_clear()


def _throttle():
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        "GetSecretValue",
    )


@patch("boto3.client")
def test_failure_then_success_returns_value_on_second_call(mock_boto3_client):
    client = Mock()
    client.get_secret_value.side_effect = [_throttle(), {"SecretString": "sa-json"}]
    mock_boto3_client.return_value = client

    assert get_secret("hairme-play-service-account") is None
    assert get_secret("hairme-play-service-account") == "sa-json"
    assert client.get_secret_value.call_count == 2


@patch("boto3.client")
def test_unexpected_exception_is_not_cached(mock_boto3_client):
    client = Mock()
    client.get_secret_value.side_effect = [
        ConnectionError("network blip"),
        {"SecretString": "value"},
    ]
    mock_boto3_client.return_value = client

    assert get_secret("s") is None
    assert get_secret("s") == "value"


@patch("boto3.client")
def test_success_is_cached(mock_boto3_client):
    client = Mock()
    client.get_secret_value.return_value = {"SecretString": "cached-value"}
    mock_boto3_client.return_value = client

    assert get_secret("s") == "cached-value"
    assert get_secret("s") == "cached-value"
    assert get_secret("s", "ap-northeast-2") == "cached-value"
    assert client.get_secret_value.call_count == 1

    # 리전이 다르면 별도 키
    assert get_secret("s", "us-east-1") == "cached-value"
    assert client.get_secret_value.call_count == 2


@patch("boto3.client")
def test_cache_clear_forces_refetch(mock_boto3_client):
    client = Mock()
    client.get_secret_value.side_effect = [
        {"SecretString": "v1"},
        {"SecretString": "v2"},
    ]
    mock_boto3_client.return_value = client

    assert get_secret("s") == "v1"
    get_secret.cache_clear()
    assert get_secret("s") == "v2"
