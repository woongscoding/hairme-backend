"""Google Play 환불/취소 구매 크레딧 회수 서비스 (voidedpurchases 폴링)

보안 감사 H2: 결제 후 환불/취소/차지백된 구매의 크레딧이 그대로 남아
"환불 + 크레딧 유지" 악용이 가능한 문제를 막는다.

동작 (EventBridge 1일 1회 → main.handler({"job": "reclaim_voided_purchases"})):
1. purchases.voidedpurchases 로 최근 N일(기본 30일, Play 최대치) 환불 목록 조회
2. 각 purchaseToken 에 대해 지급 당시 기록된 클레임 마커
   (ledger 테이블의 user_id="purchase#<token>", sk="claim")로 사용자/상품 해석
3. 회수 멱등 마커 "void#<token>" 를 조건부 put 으로 선점한 뒤
   credit_service.consume_for_reclaim() 으로 차감 (잔액 음수 허용)

지급 대상/금액 해석 순서 (원장에는 ref_id GSI가 없어 토큰 → 사용자 역추적 필요):
- user_id: 클레임 마커의 claimed_by (구매 시 credits.py 가 기록)
- amount : 마커 detail.product_id → settings.CREDIT_PRODUCTS 매핑
           (상품 매핑이 바뀐 과거 구매는) → 해당 사용자 원장에서
           reason="purchase" AND ref_id=<token> 항목의 amount 로 폴백
- 마커가 없으면(테스트 결제/타 경로 지급 등) not_found 로 집계하고 건너뛴다
"""

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from config.settings import settings
from core.logging import log_structured, logger
from services.credit_service import get_credit_service
from services.play_billing_service import PlayBillingError, get_play_billing_service

# 페이지네이션 무한 루프 방지 (1페이지 기본 1000건 → 충분)
MAX_PAGES = 50


def _mask_token(token: str) -> str:
    """구매 토큰은 원문 로깅 금지 - sha256 앞 12자리만 남긴다"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


class PlayVoidReclaimService:
    """환불된 Play 구매의 크레딧 회수 (일 1회 배치)"""

    def __init__(self, play_service=None, credit_service=None):
        self._play_service = play_service
        self._credit_service = credit_service

    @property
    def play_service(self):
        if self._play_service is None:
            self._play_service = get_play_billing_service()
        return self._play_service

    @property
    def credit_service(self):
        if self._credit_service is None:
            self._credit_service = get_credit_service()
        return self._credit_service

    def run(self, lookback_days: Optional[int] = None) -> Dict[str, Any]:
        """
        환불 목록 폴링 → 크레딧 회수

        배치 잡이므로 어떤 예외도 밖으로 던지지 않는다.

        Returns:
            {"checked", "reclaimed", "already_reclaimed", "not_found",
             "failed", "disabled"} 요약
        """
        if lookback_days is None:
            lookback_days = settings.PLAY_VOID_LOOKBACK_DAYS

        summary: Dict[str, Any] = {
            "checked": 0,
            "reclaimed": 0,
            "already_reclaimed": 0,
            "not_found": 0,
            "failed": 0,
            "disabled": False,
        }

        disabled = self._check_configuration()
        if disabled is not None:
            summary.update(disabled)
            return summary

        start_time_millis = int(
            (
                datetime.now(timezone.utc) - timedelta(days=int(lookback_days))
            ).timestamp()
            * 1000
        )

        page_token: Optional[str] = None
        for _ in range(MAX_PAGES):
            try:
                page = self.play_service.list_voided_purchases(
                    start_time_millis, page_token=page_token
                )
            except Exception as e:
                # 401/500 등 - 다음 실행에서 재시도되므로 실패만 집계하고 종료
                summary["failed"] += 1
                logger.error(f"❌ voidedpurchases 조회 실패 (중단): {str(e)}")
                break

            for entry in page.get("voidedPurchases") or []:
                self._process_entry(entry, summary)

            page_token = (page.get("tokenPagination") or {}).get("nextPageToken")
            if not page_token:
                break
        else:
            logger.warning(
                f"⚠️ voidedpurchases 최대 페이지({MAX_PAGES}) 도달 - 일부 미처리"
            )

        logger.info(
            "🧾 Play 환불 회수 완료: "
            f"checked={summary['checked']}, reclaimed={summary['reclaimed']}, "
            f"already={summary['already_reclaimed']}, "
            f"not_found={summary['not_found']}, failed={summary['failed']}"
        )
        return summary

    # ---------- 내부 ----------

    def _check_configuration(self) -> Optional[Dict[str, Any]]:
        """설정 누락이면 disabled 요약, 정상이면 None (fail-safe)"""
        try:
            if not self.play_service.package_name:
                logger.warning("⚠️ PLAY_PACKAGE_NAME 미설정 - 환불 회수 잡을 건너뜁니다")
                return {"disabled": True, "reason": "package_name_not_configured"}

            # 자격증명 로드 실패(시크릿 없음/파싱 실패)를 여기서 확정
            _ = self.play_service.session
        except PlayBillingError as e:
            logger.warning(f"⚠️ Play 자격증명 사용 불가 - 환불 회수 잡 건너뜀: {str(e)}")
            return {"disabled": True, "reason": "credentials_unavailable"}
        except Exception as e:
            logger.warning(f"⚠️ 환불 회수 잡 초기화 실패 - 건너뜀: {str(e)}")
            return {"disabled": True, "reason": "initialization_failed"}
        return None

    def _process_entry(self, entry: Dict[str, Any], summary: Dict[str, Any]) -> None:
        """환불 1건 처리 (예외는 failed 집계 후 흡수)"""
        token = entry.get("purchaseToken")
        if not token:
            summary["failed"] += 1
            logger.warning("⚠️ purchaseToken 없는 환불 항목 - 건너뜀")
            return

        summary["checked"] += 1
        token_hash = _mask_token(token)
        base_log = {
            "token_hash": token_hash,
            "order_id": entry.get("orderId"),
            "voided_time_millis": entry.get("voidedTimeMillis"),
            "voided_source": entry.get("voidedSource"),
            "voided_reason": entry.get("voidedReason"),
        }

        try:
            resolved = self._resolve_grant(token)
            if resolved is None:
                summary["not_found"] += 1
                log_structured("play_void_reclaim", {**base_log, "result": "not_found"})
                return

            user_id, amount = resolved
            void_ref = f"void#{token}"

            if not self.credit_service.try_claim_ref(
                void_ref,
                user_id,
                detail={"order_id": entry.get("orderId"), "amount": amount},
            ):
                summary["already_reclaimed"] += 1
                log_structured(
                    "play_void_reclaim",
                    {**base_log, "result": "already_reclaimed", "user_id": user_id},
                )
                return

            try:
                balance_after = self.credit_service.consume_for_reclaim(
                    user_id, amount, ref_id=token
                )
            except ValueError:
                # 탈퇴 등으로 사용자가 없음 - 회수 대상이 없으므로 마커는 유지
                summary["not_found"] += 1
                log_structured(
                    "play_void_reclaim",
                    {**base_log, "result": "user_missing", "user_id": user_id},
                )
                return
            except Exception:
                # 회수 실패 - 다음 실행에서 재시도할 수 있게 마커 회수
                self.credit_service.release_ref(void_ref)
                raise

            summary["reclaimed"] += 1
            log_structured(
                "play_void_reclaim",
                {
                    **base_log,
                    "result": "reclaimed",
                    "user_id": user_id,
                    "amount": amount,
                    "balance_after": balance_after,
                },
            )
        except Exception as e:
            summary["failed"] += 1
            logger.error(
                f"❌ 환불 크레딧 회수 실패: token_hash={token_hash}, error={str(e)}",
                exc_info=True,
            )

    def _resolve_grant(self, token: str) -> Optional[Tuple[str, int]]:
        """구매 토큰 → (user_id, 지급했던 크레딧) 역추적 (없으면 None)"""
        marker = self.credit_service.get_claim(f"purchase#{token}")
        if not marker:
            return None

        user_id = marker.get("claimed_by")
        if not user_id:
            return None

        detail = marker.get("detail") or {}
        product_id = detail.get("product_id")
        amount = settings.CREDIT_PRODUCTS.get(product_id) if product_id else None

        if not amount:
            # 상품 매핑이 없거나 바뀐 과거 구매 - 원장 기록으로 폴백
            entry = self.credit_service.find_ledger_entry(
                user_id, token, reason="purchase"
            )
            if entry is not None:
                amount = abs(int(entry.get("amount", 0)))

        if not amount or int(amount) <= 0:
            logger.warning(
                f"⚠️ 환불 구매의 지급 크레딧을 확인할 수 없음: "
                f"token_hash={_mask_token(token)}"
            )
            return None

        return str(user_id), int(amount)


# Singleton
_play_void_reclaim_service: Optional[PlayVoidReclaimService] = None


def get_play_void_reclaim_service() -> PlayVoidReclaimService:
    global _play_void_reclaim_service
    if _play_void_reclaim_service is None:
        _play_void_reclaim_service = PlayVoidReclaimService()
    return _play_void_reclaim_service
