"""무료 정책 실험 코호트 (수익화 v2.1 Phase 1)

신규 기기를 해시로 50:50 나눠 실험군에는 더 좁은 무료 정책을 적용한다.

| 구분          | 대조군 (control, grandfathered) | 실험군 (treatment)          |
| ------------- | ------------------------------- | --------------------------- |
| 비회원 합성   | 매일 DAILY_SYNTHESIS_LIMIT 회   | 평생 TREATMENT_GUEST_LIFETIME_LIMIT 회 |
| 가입 보너스   | SIGNUP_BONUS_CREDITS            | TREATMENT_SIGNUP_BONUS_CREDITS |
| 보상형 광고   | 크레딧 1 (REWARD_AD_GRANTS_CREDIT) | 원가 0원 기능 해제        |

코호트는 기기를 처음 본 순간 한 번 정해져 사용량 테이블(cohort#<device_id>)에
남고 바뀌지 않는다. 처음 본 시각이 실험 시작 전이면 grandfathered(대조군 고정).
회원은 가입할 때 기기의 코호트를 계정(free_policy_arm)에 복사한다.

플래그를 내리면(FREE_POLICY_EXPERIMENT_ENABLED=False) 기록된 코호트와 무관하게
전원 대조군 정책이 적용된다 (= 즉시 원복). 기록은 분석용으로 남는다.

코호트 저장소 장애 시에는 대조군으로 처리한다 (사용자에게 더 너그러운 쪽).
"""

import hashlib
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import settings
from core.logging import log_structured, logger, mask_user_id

ARM_CONTROL = "control"
ARM_TREATMENT = "treatment"
ARM_GRANDFATHERED = "grandfathered"

# 보상형 광고 1회로 당일 자정(KST)까지 해제되는 원가 0원 기능.
# 서버는 해제 상태만 알려주고, 화면 노출은 앱이 이 이름으로 판단한다.
REWARD_UNLOCK_FEATURES = ("personal_color_detail", "full_style_ranking")


def experiment_start() -> Optional[datetime]:
    """FREE_POLICY_EXPERIMENT_START 파싱 (비었거나 형식 오류면 None)"""
    raw = (settings.FREE_POLICY_EXPERIMENT_START or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        logger.error(f"FREE_POLICY_EXPERIMENT_START 형식 오류: {raw!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def hash_bucket(subject: str) -> int:
    """subject 를 0~99 버킷으로 (솔트 포함 sha256 - 솔트를 바꾸면 재배정)"""
    digest = hashlib.sha256(
        f"{settings.FREE_POLICY_SALT}:{subject}".encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16) % 100


def arm_for_new_device(device_id: str, now: Optional[datetime] = None) -> str:
    """처음 보는 기기에 줄 코호트"""
    start = experiment_start()
    now = now or datetime.now(timezone.utc)
    if not settings.FREE_POLICY_EXPERIMENT_ENABLED or start is None or now < start:
        return ARM_GRANDFATHERED
    if hash_bucket(device_id) < settings.FREE_POLICY_TREATMENT_PERCENT:
        return ARM_TREATMENT
    return ARM_CONTROL


def effective_arm(arm: Optional[str]) -> str:
    """정책 적용용 코호트. 실험이 꺼져 있으면 누구든 대조군"""
    if settings.FREE_POLICY_EXPERIMENT_ENABLED and arm == ARM_TREATMENT:
        return ARM_TREATMENT
    return ARM_CONTROL


def resolve_device_arm(usage_service: Any, device_id: str) -> str:
    """기기 코호트 조회(처음이면 등록). device_id 는 검증된 값이어야 한다.

    반환값은 기록된 원본 코호트(control/treatment/grandfathered)다.
    정책에 쓸 때는 effective_arm() 을 거칠 것.
    """
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    candidate = arm_for_new_device(device_id, now)
    try:
        cohort = usage_service.register_cohort(
            f"cohort#{device_id}", candidate, now_iso
        )
    except Exception as e:
        logger.error(f"⚠️ 코호트 조회/등록 실패 - 대조군 처리: {str(e)}")
        return ARM_CONTROL

    arm = cohort["arm"]
    if cohort["first_seen"] == now_iso:
        # 실험 판정의 분모 (코호트별 신규 기기 수)
        log_structured(
            "free_policy_assigned", {"device_id": mask_user_id(device_id), "arm": arm}
        )
    return arm


def signup_bonus_for(arm: Optional[str]) -> int:
    if effective_arm(arm) == ARM_TREATMENT:
        return settings.TREATMENT_SIGNUP_BONUS_CREDITS
    return settings.SIGNUP_BONUS_CREDITS


def reward_grants_credit(arm: Optional[str]) -> bool:
    """보상형 광고 보상이 크레딧인지 (False 면 원가 0원 기능 해제)"""
    if effective_arm(arm) == ARM_TREATMENT:
        return False
    return settings.REWARD_AD_GRANTS_CREDIT
