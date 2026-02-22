"""
A/B 테스트 라우터 모듈

새로 학습된 ML 모델의 실제 성능을 검증하기 위해
사용자 트래픽을 Champion(기존)과 Challenger(신규) 모델로 분배합니다.

핵심 기능:
- 해시 기반 일관된 변형 할당 (동일 user_id → 동일 모델)
- 환경변수 기반 on/off 및 트래픽 비율 조정
- S3에서 모델 로드 (Lazy Loading)

Author: HairMe ML Team
Date: 2025-12-02
Version: 1.0.0
"""

import os
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelVariant(Enum):
    """A/B 테스트 모델 변형"""

    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass
class ABTestConfig:
    """
    A/B 테스트 설정

    Attributes:
        experiment_id: 실험 ID (예: "exp_2025_12_02")
        champion_model_version: 현재 모델 버전 (예: "v6_20251201")
        challenger_model_version: 새 모델 버전 (예: "v6_20251202")
        challenger_traffic_percent: 새 모델 트래픽 비율 (0-100)
        enabled: 실험 활성화 여부
    """

    experiment_id: str = ""
    champion_model_version: str = "v6"
    challenger_model_version: str = ""
    challenger_traffic_percent: int = 10
    enabled: bool = False
    started_at: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ABTestConfig":
        """환경변수에서 설정 로드"""
        enabled = os.getenv("ABTEST_ENABLED", "false").lower() == "true"

        return cls(
            experiment_id=os.getenv("ABTEST_EXPERIMENT_ID", ""),
            champion_model_version=os.getenv("ABTEST_CHAMPION_VERSION", "v6"),
            challenger_model_version=os.getenv("ABTEST_CHALLENGER_VERSION", ""),
            challenger_traffic_percent=int(
                os.getenv("ABTEST_CHALLENGER_PERCENT", "10")
            ),
            enabled=enabled,
            started_at=os.getenv("ABTEST_STARTED_AT", None),
        )

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "experiment_id": self.experiment_id,
            "champion_model_version": self.champion_model_version,
            "challenger_model_version": self.challenger_model_version,
            "challenger_traffic_percent": self.challenger_traffic_percent,
            "enabled": self.enabled,
            "started_at": self.started_at,
        }


class ABTestRouter:
    """
    A/B 테스트 라우터

    해시 기반으로 사용자를 Champion 또는 Challenger 모델로 일관되게 라우팅합니다.

    Usage:
        router = ABTestRouter()
        variant = router.get_variant(user_id="user_123")
        model_version = router.get_model_version(variant)
    """

    def __init__(self, config: Optional[ABTestConfig] = None):
        """
        초기화

        Args:
            config: A/B 테스트 설정 (None이면 환경변수에서 로드)
        """
        self.config = config or ABTestConfig.from_env()

        if self.config.enabled:
            logger.info(
                f"🔬 A/B 테스트 활성화: experiment={self.config.experiment_id}, "
                f"challenger_traffic={self.config.challenger_traffic_percent}%"
            )
        else:
            logger.info("🔬 A/B 테스트 비활성화 - Champion 모델만 사용")

    def get_variant(self, user_id: str) -> ModelVariant:
        """
        사용자 ID 기반으로 모델 변형 결정

        해시 기반이므로 동일한 user_id는 항상 동일한 모델을 받습니다.

        Args:
            user_id: 사용자 ID (analysis_id도 가능)

        Returns:
            ModelVariant.CHAMPION or ModelVariant.CHALLENGER
        """
        # A/B 테스트 비활성화 시 항상 Champion
        if not self.config.enabled:
            return ModelVariant.CHAMPION

        # Challenger 모델 버전이 없으면 Champion
        if not self.config.challenger_model_version:
            logger.warning("⚠️ Challenger 모델 버전이 설정되지 않음 - Champion 사용")
            return ModelVariant.CHAMPION

        # 해시 기반 분배 (0-99 범위)
        hash_value = self._get_hash_bucket(user_id)

        # challenger_traffic_percent% 확률로 Challenger
        if hash_value < self.config.challenger_traffic_percent:
            logger.debug(f"[ABTEST] user={user_id} -> CHALLENGER (hash={hash_value})")
            return ModelVariant.CHALLENGER
        else:
            logger.debug(f"[ABTEST] user={user_id} -> CHAMPION (hash={hash_value})")
            return ModelVariant.CHAMPION

    def get_model_version(self, variant: ModelVariant) -> str:
        """
        변형에 해당하는 모델 버전 반환

        Args:
            variant: ModelVariant.CHAMPION or ModelVariant.CHALLENGER

        Returns:
            모델 버전 문자열 (예: "v6_20251202")
        """
        if variant == ModelVariant.CHALLENGER:
            return self.config.challenger_model_version
        return self.config.champion_model_version

    def get_experiment_info(self, variant: ModelVariant) -> Dict[str, Any]:
        """
        실험 정보 반환 (피드백 저장용)

        Args:
            variant: 선택된 모델 변형

        Returns:
            {
                'experiment_id': 'exp_2025_12_02',
                'model_version': 'v6_20251202',
                'ab_variant': 'challenger'
            }
        """
        return {
            "experiment_id": self.config.experiment_id if self.config.enabled else "",
            "model_version": self.get_model_version(variant),
            "ab_variant": variant.value,
        }

    def _get_hash_bucket(self, user_id: str) -> int:
        """
        사용자 ID를 0-99 범위의 버킷으로 변환

        MD5 해시를 사용하여 일관된 분배를 보장합니다.

        Args:
            user_id: 사용자 ID

        Returns:
            0-99 범위의 정수
        """
        # 실험 ID를 salt로 사용하여 실험별 다른 분배
        salt = self.config.experiment_id or "default"
        hash_input = f"{salt}:{user_id}"

        # MD5 해시 계산
        hash_bytes = hashlib.md5(hash_input.encode()).digest()

        # 첫 4바이트를 정수로 변환 후 0-99 범위로 모듈러
        hash_int = int.from_bytes(hash_bytes[:4], byteorder="big")
        return hash_int % 100

    def is_abtest_active(self) -> bool:
        """A/B 테스트 활성화 여부"""
        return self.config.enabled and bool(self.config.challenger_model_version)

    def update_config(self, new_config: ABTestConfig) -> None:
        """
        런타임에 설정 업데이트

        Note: 이 변경은 메모리에서만 유효하며, 재시작 시 환경변수에서 다시 로드됨
        """
        self.config = new_config
        logger.info(f"🔧 A/B 테스트 설정 업데이트: {new_config.to_dict()}")


# ========== 싱글톤 인스턴스 ==========
_ab_router_instance: Optional[ABTestRouter] = None


def get_ab_router() -> ABTestRouter:
    """
    A/B 테스트 라우터 싱글톤 인스턴스 반환

    Returns:
        ABTestRouter 인스턴스
    """
    global _ab_router_instance

    if _ab_router_instance is None:
        _ab_router_instance = ABTestRouter()

    return _ab_router_instance


def refresh_ab_router() -> ABTestRouter:
    """
    A/B 테스트 라우터 재초기화 (설정 변경 시)

    환경변수가 변경된 후 호출하면 새 설정이 적용됩니다.

    Returns:
        새로운 ABTestRouter 인스턴스
    """
    global _ab_router_instance
    _ab_router_instance = ABTestRouter()
    return _ab_router_instance
