"""서빙(models/ml_recommender)과 트레이너(lambda_trainer) 모델 정의 일치 검사

트레이너는 독립 이미지라 저장소 코드를 import 하지 않는다 (test_trainer_has_no_repo_imports).
그래서 RecommendationModelV6 가 두 곳에 따로 정의돼 있는데, 한쪽만 고치면
트레이너가 만든 체크포인트를 서빙이 못 읽거나(키 불일치) 같은 가중치로 다른 점수를 낸다.
공유 패키지 대신 이 테스트로 두 정의가 어긋나는 순간 CI 에서 막는다.
"""

import sys

import pytest
import torch

import models.ml_recommender as serving

BUNDLED_MODEL = "models/hairstyle_recommender_v6_multitoken.pt"


def _import_trainer():
    sys.path.insert(0, "lambda_trainer")
    try:
        import lambda_function

        return lambda_function
    finally:
        sys.path.pop(0)


trainer = _import_trainer()


def _pair():
    torch.manual_seed(0)
    serve_model = serving.RecommendationModelV6()
    train_model = trainer.RecommendationModelV6()
    return serve_model, train_model


def test_state_dict_keys_and_shapes_match():
    serve_model, train_model = _pair()
    serve_sd = serve_model.state_dict()
    train_sd = train_model.state_dict()

    assert serve_sd.keys() == train_sd.keys()
    for key in serve_sd:
        assert serve_sd[key].shape == train_sd[key].shape, key


def test_same_weights_give_same_scores():
    serve_model, train_model = _pair()
    train_model.load_state_dict(serve_model.state_dict(), strict=True)
    serve_model.eval()
    train_model.eval()

    torch.manual_seed(1)
    face = torch.randn(8, 6)
    skin = torch.randn(8, 2)
    style = torch.randn(8, 384)

    with torch.no_grad():
        a = serve_model(face, skin, style)
        b = train_model(face, skin, style)

    assert torch.allclose(a, b, atol=1e-6)


def test_bundled_checkpoint_loads_into_trainer_strictly():
    """트레이너 from_base 가 쓰는 번들 v6 체크포인트를 트레이너 클래스가 그대로 읽는다"""
    try:
        checkpoint = torch.load(BUNDLED_MODEL, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        pytest.skip("번들 모델 파일 없음")

    state = checkpoint.get("model_state_dict", checkpoint)
    model = trainer.RecommendationModelV6()
    model.load_state_dict(state, strict=True)
