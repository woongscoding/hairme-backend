"""
추천 Top-3 동점권 탐색 (explore_v1)

모델 점수 표준편차가 2~5점(10~95 척도)이라 상위 10~20개가 사실상 동점인데,
정렬이 결정적이라 미세한 차이가 매번 같은 승자를 만든다
(2026-09 실측: 1위 상위 3종 94%, 시스루 댄디 단독 65%).

동점권(top - margin 이상) 안에서 softmax(score / T) 가중치로 비복원 추출해
노출을 분산하고, 각 카드의 포함 확률(propensity)을 남긴다.
propensity 는 이후 암묵 피드백 재학습에서 역확률 가중(IPS)으로 순위 편향을
보정하는 데 쓴다.

모델·AWS 의존이 없는 순수 함수만 둔다. 난수는 image_hash + model_version 시드의
전용 Generator 만 쓰고 전역 random / np.random 상태는 건드리지 않는다
(같은 사진을 다시 분석하면 같은 결과).
"""

import hashlib
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

POLICY_DETERMINISTIC = "deterministic"
POLICY_EXPLORE = "explore_v1"

ARM_EXPLORE = "explore"
ARM_CONTROL = "control"

# hairme-analysis 의 experiment_id (ab_variant 에 arm 을 저장)
EXPERIMENT_ID = "rec_explore_v1"

# 솔트를 바꾸면 전원 재배정된다 (무료 정책 실험과 독립이 되도록 별도 솔트)
ARM_SALT = "rec-explore-v1"

# 포함 확률 몬테카를로 횟수 (풀 20개 이하에서 수 ms, 표준오차 약 0.01)
PROPENSITY_SIMULATIONS = 2000

# 경계값(top - margin) 비교의 부동소수 오차 허용
_EPS = 1e-9


@dataclass(frozen=True)
class ExploreSelection:
    """explore_select 결과

    indices: 선택된 후보의 풀 인덱스 (점수 내림차순 = 오름차순 인덱스)
    propensities: indices 와 같은 순서의 포함 확률
    band_size: 동점권 크기 (3개 미만이면 점수순으로 채운 크기)
    """

    indices: List[int]
    propensities: List[float]
    band_size: int


def assignment_bucket(user_id: Optional[str], image_hash: str) -> float:
    """실험 배정 버킷 [0, 1). 로그인 사용자는 user_id, 아니면 image_hash 기준"""
    subject = f"user:{user_id}" if user_id else f"image:{image_hash}"
    digest = hashlib.sha256(f"{ARM_SALT}:{subject}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_arm(user_id: Optional[str], image_hash: str, ratio: float) -> str:
    """버킷이 ratio 미만이면 실험군, 아니면 대조군"""
    if assignment_bucket(user_id, image_hash) < ratio:
        return ARM_EXPLORE
    return ARM_CONTROL


def selection_seed(image_hash: str, model_version: str) -> int:
    """sha256(image_hash + model_version) 앞 8바이트 - 모델이 바뀌면 재추첨"""
    digest = hashlib.sha256(f"{image_hash}{model_version}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def tie_band_size(scores: Sequence[float], margin: float, k: int = 3) -> int:
    """동점권 크기. scores 는 내림차순 원점수.

    top - margin 이상(경계 포함)인 후보 수. k개 미만이면 점수순으로 k개까지 채운다.
    """
    if not scores:
        return 0
    threshold = scores[0] - margin - _EPS
    within = sum(1 for s in scores if s >= threshold)
    return max(within, min(k, len(scores)))


def _logits(scores: Sequence[float], temperature: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64)
    # 최댓값을 빼도 softmax 는 같다 (오버플로 방지)
    return (arr - arr.max()) / temperature


def inclusion_probabilities(
    scores: Sequence[float],
    k: int,
    temperature: float,
    seed: int,
    simulations: int = PROPENSITY_SIMULATIONS,
) -> List[float]:
    """softmax(score / T) 비복원 k개 추출에서 각 후보가 뽑힐 확률 (몬테카를로)

    Gumbel-top-k: logits + Gumbel 노이즈의 상위 k개가 비복원 softmax 추출과 같은
    분포다. 합은 정확히 min(k, n) 이다.
    """
    n = len(scores)
    if n <= k:
        return [1.0] * n
    rng = np.random.default_rng((seed, 1))
    keys = _logits(scores, temperature) + rng.gumbel(size=(simulations, n))
    top = np.argpartition(-keys, k - 1, axis=1)[:, :k]
    counts = np.bincount(top.ravel(), minlength=n)
    return (counts / simulations).tolist()


def explore_select(
    scores: Sequence[float],
    k: int,
    margin: float,
    temperature: float,
    seed: int,
) -> ExploreSelection:
    """동점권 안에서 k개를 확률 선택한다. scores 는 내림차순 원점수(풀 순서)."""
    band = tie_band_size(scores, margin, k)
    if band <= k:
        return ExploreSelection(list(range(band)), [1.0] * band, band)

    band_scores = list(scores[:band])
    rng = np.random.default_rng((seed, 0))
    keys = _logits(band_scores, temperature) + rng.gumbel(size=band)
    # 표시는 뽑힌 순서가 아니라 점수 내림차순 (풀이 내림차순이므로 인덱스 오름차순)
    chosen = sorted(int(i) for i in np.argsort(-keys)[:k])

    probs = inclusion_probabilities(band_scores, k, temperature, seed)
    # 뽑힌 카드의 진짜 확률은 0보다 크다 - 시뮬레이션에서 0이 나와도 하한을 둔다
    floor = 1.0 / PROPENSITY_SIMULATIONS
    propensities = [round(max(probs[i], floor), 4) for i in chosen]
    return ExploreSelection(chosen, propensities, band)
