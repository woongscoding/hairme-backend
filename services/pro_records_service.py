"""Pro 계정(미용실) 손님 기록 서비스

디자이너가 손님(이름·전화번호·메모)과 상담 기록(날짜·성별·합성한 스타일과 결과 사진·고른 스타일)을
서버에 보관한다. 저장 위치:
- DynamoDB: database/pro_customer_repository.py (손님·전화번호 마커·상담 기록)
- S3: pro/records/{user_id}/{customer_id}/{visit_id}/{result_id}.{jpg|png}
  (services/photo_storage_service.py)

보관 기간: 상담 기록과 사진은 상담일부터 2년, 손님은 마지막 상담일부터 2년
(상담이 없으면 등록일부터). DynamoDB TTL 과 S3 수명주기 규칙이 지우고,
그 전에라도 기간이 지난 기록은 읽을 때 빼고 보여 준다.

삭제 순서는 항상 사진(S3) → 기록(DynamoDB) 이다. 사진 삭제가 실패하면 기록을 남겨
앱이 다시 시도할 수 있게 한다 (기록 없이 사진만 남지 않게).

이름·전화번호·메모는 로그에 남기지 않는다.
"""

import re
import secrets
from typing import Any, Dict, List, Optional

from config.settings import settings
from core.logging import log_structured, logger, mask_user_id
from database.pro_customer_repository import (
    CUSTOMER_PREFIX,
    CustomerGoneError,
    DuplicatePhoneError,
    ProCustomerRepository,
    RecordsTableMissing,
    VISIT_PREFIX,
    VisitChangedError,
    get_pro_customer_repository,
    now_utc,
    parse_iso,
)
from services.photo_storage_service import (
    PhotoStorageService,
    get_photo_storage_service,
    pro_record_photo_key,
    pro_records_prefix,
)

# ===== 입력 규칙 (앱과 맞춘 값) =====
NAME_MAX_LENGTH = 20
MEMO_MAX_LENGTH = 500
STYLE_MAX_LENGTH = 50
PHONE_PATTERN = re.compile(r"0\d{8,10}")
# 서버가 만드는 손님 id 와 앱이 만드는 상담 id 모두 이 모양만 받는다 (S3 경로에 들어간다)
ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,40}")
GENDERS = ("female", "male")

# 상담 1건에 올릴 수 있는 결과 수
MAX_RESULTS_PER_VISIT = 30
# 상담 사진 최대 크기
MAX_PHOTO_BYTES = 3 * 1024 * 1024

# secrets.token_urlsafe(n) → 약 4n/3 자
CUSTOMER_ID_BYTES = 12  # 16자
RESULT_ID_BYTES = 9  # 12자

# 동시에 같은 상담에 결과를 올리는 경우 다시 읽고 시도하는 횟수
_ADD_RESULT_ATTEMPTS = 3

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_CONTROL_CHARS_EXCEPT_NEWLINE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f]")


# ========== 오류 ==========


class RecordsUnavailable(Exception):
    """테이블·버킷이 설정되지 않았거나 아직 만들어지지 않았다"""


class InvalidCustomerInput(Exception):
    """이름·전화번호·메모 형식이 맞지 않는다"""


class CustomerNotFound(Exception):
    pass


class VisitNotFound(Exception):
    pass


class DuplicatePhone(Exception):
    """이 디자이너의 다른 손님이 같은 번호를 쓰고 있다 (customer = 그 손님의 응답 모양)"""

    def __init__(self, customer: Dict[str, Any]):
        self.customer = customer
        super().__init__("이미 등록된 전화번호입니다")


class TooManyResults(Exception):
    pass


class InvalidStyle(Exception):
    """고른 스타일이 그 상담의 결과에 없다"""


class SaveFailed(Exception):
    """사진 또는 기록 저장 실패"""


class DeleteFailed(Exception):
    """사진 또는 기록 삭제 실패 (남은 기록은 그대로)"""


# ========== 입력 정리 ==========


def normalize_name(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidCustomerInput()
    name = value.strip()
    if not name or len(name) > NAME_MAX_LENGTH or _CONTROL_CHARS.search(name):
        raise InvalidCustomerInput()
    return name


def normalize_phone(value: Any) -> str:
    """숫자만 남긴다 (010-1234-5678 → 01012345678)"""
    if not isinstance(value, str):
        raise InvalidCustomerInput()
    phone = re.sub(r"\D", "", value)
    if not PHONE_PATTERN.fullmatch(phone):
        raise InvalidCustomerInput()
    return phone


def normalize_memo(value: Any) -> str:
    """줄바꿈은 허용한다. 앞뒤 공백만 정리한다"""
    if not isinstance(value, str):
        raise InvalidCustomerInput()
    memo = value.strip()
    if len(memo) > MEMO_MAX_LENGTH or _CONTROL_CHARS_EXCEPT_NEWLINE.search(memo):
        raise InvalidCustomerInput()
    return memo


def normalize_style(value: Any) -> Optional[str]:
    """스타일 이름 (1~50자, 제어 문자 없음). 맞지 않으면 None"""
    if not isinstance(value, str):
        return None
    style = value.strip()
    if not style or len(style) > STYLE_MAX_LENGTH or _CONTROL_CHARS.search(style):
        return None
    return style


def is_valid_id(value: Any) -> bool:
    return isinstance(value, str) and bool(ID_PATTERN.fullmatch(value))


def detect_record_photo(data: bytes) -> Optional[str]:
    """매직 바이트로 jpeg / png 판별 (그 밖은 None)"""
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    return None


# ========== 응답 모양 ==========


def public_customer(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "customer_id": item["sk"][len(CUSTOMER_PREFIX) :],
        "name": item.get("name", ""),
        "phone": item.get("phone", ""),
        "memo": item.get("memo") or "",
        "created_at": item.get("created_at"),
        "last_visit_at": item.get("last_visit_at") or None,
        "visit_count": int(item.get("visit_count") or 0),
    }


def _activity_key(item: Dict[str, Any]):
    moment = parse_iso(item.get("last_visit_at")) or parse_iso(item.get("created_at"))
    return (moment.timestamp() if moment else 0.0, item.get("sk", ""))


class ProRecordsService:
    """손님 기록 기능 (엔드포인트와 회원 탈퇴가 쓴다)"""

    def __init__(
        self,
        repository: Optional[ProCustomerRepository] = None,
        storage: Optional[PhotoStorageService] = None,
    ):
        self._repository = repository
        self._storage = storage

    @property
    def repository(self) -> ProCustomerRepository:
        return self._repository or get_pro_customer_repository()

    @property
    def storage(self) -> PhotoStorageService:
        return self._storage or get_photo_storage_service()

    @property
    def available(self) -> bool:
        return self.repository.enabled and self.storage.enabled

    def _require_available(self) -> None:
        if not self.available:
            raise RecordsUnavailable()

    def _public_visit(self, item: Dict[str, Any]) -> Dict[str, Any]:
        customer_id, _, visit_id = item["sk"][len(VISIT_PREFIX) :].partition("#")
        results = []
        for result in item.get("results") or []:
            results.append(
                {
                    "result_id": result.get("result_id"),
                    "style": result.get("style"),
                    "photo_url": self.storage.presigned_url(
                        result.get("photo_key") or "",
                        settings.PRO_RECORD_PHOTO_URL_SECONDS,
                    ),
                }
            )
        return {
            "visit_id": visit_id,
            "customer_id": customer_id,
            "at": item.get("at"),
            "gender": item.get("gender"),
            "chosen_style": item.get("chosen_style") or None,
            "results": results,
        }

    def _customer(self, user_id: str, customer_id: str) -> Dict[str, Any]:
        if not is_valid_id(customer_id):
            raise CustomerNotFound()
        customer = self.repository.get_customer(user_id, customer_id, now_utc())
        if customer is None:
            raise CustomerNotFound()
        return customer

    # ========== 손님 ==========

    def list_customers(self, user_id: str) -> List[Dict[str, Any]]:
        """최근 활동(마지막 상담일, 없으면 등록일) 순"""
        self._require_available()
        items = self.repository.list_customers(user_id, now_utc())
        items.sort(key=_activity_key, reverse=True)
        return [public_customer(item) for item in items]

    def create_customer(self, user_id: str, name: Any, phone: Any) -> Dict[str, Any]:
        self._require_available()
        name = normalize_name(name)
        phone = normalize_phone(phone)
        try:
            item = self.repository.create_customer(
                user_id,
                secrets.token_urlsafe(CUSTOMER_ID_BYTES),
                name,
                phone,
                now_utc(),
            )
        except DuplicatePhoneError as e:
            raise DuplicatePhone(public_customer(e.customer)) from e
        log_structured("pro_customer_created", {"user_id": mask_user_id(user_id)})
        return public_customer(item)

    def update_customer(
        self, user_id: str, customer_id: str, body: Dict[str, Any]
    ) -> Dict[str, Any]:
        self._require_available()
        changes: Dict[str, str] = {}
        if "name" in body:
            changes["name"] = normalize_name(body["name"])
        if "phone" in body:
            changes["phone"] = normalize_phone(body["phone"])
        if "memo" in body:
            changes["memo"] = normalize_memo(body["memo"])

        customer = self._customer(user_id, customer_id)
        try:
            item = self.repository.update_customer(
                user_id, customer, changes, now_utc()
            )
        except DuplicatePhoneError as e:
            raise DuplicatePhone(public_customer(e.customer)) from e
        except CustomerGoneError as e:
            raise CustomerNotFound() from e
        return public_customer(item)

    def delete_customer(self, user_id: str, customer_id: str) -> None:
        """손님과 상담 기록·사진 전부 삭제. 없는 손님이어도 성공 (다시 눌러도 같은 결과)"""
        self._require_available()
        if not is_valid_id(customer_id):
            return
        try:
            deleted_photos = self.storage.delete_pro_records(
                pro_records_prefix(user_id, customer_id)
            )
        except Exception as e:
            logger.error(
                f"❌ 손님 사진 삭제 실패 (기록 유지): user_id={mask_user_id(user_id)}, "
                f"{type(e).__name__}"
            )
            raise DeleteFailed() from e

        try:
            existed = self.repository.delete_customer_items(user_id, customer_id)
        except RecordsTableMissing:
            raise
        except Exception as e:
            logger.error(
                f"❌ 손님 기록 삭제 실패: user_id={mask_user_id(user_id)}, "
                f"{type(e).__name__}"
            )
            raise DeleteFailed() from e

        log_structured(
            "pro_customer_deleted",
            {
                "user_id": mask_user_id(user_id),
                "existed": existed,
                "deleted_photos": deleted_photos,
            },
        )

    # ========== 상담 기록 ==========

    def list_visits(self, user_id: str, customer_id: str) -> List[Dict[str, Any]]:
        """최근 상담 순"""
        self._require_available()
        self._customer(user_id, customer_id)
        items = self.repository.list_visits(user_id, customer_id, now_utc())

        def _key(item):
            moment = parse_iso(item.get("at"))
            return (moment.timestamp() if moment else 0.0, item.get("sk", ""))

        items.sort(key=_key, reverse=True)
        return [self._public_visit(item) for item in items]

    def add_result(
        self,
        user_id: str,
        customer_id: str,
        visit_id: str,
        style: str,
        gender: str,
        image_bytes: bytes,
        image_format: str,
    ) -> Dict[str, Any]:
        """상담 결과 1건 저장 (상담이 없으면 만든다). 사진을 먼저 저장하고 기록을 쓴다.

        Raises:
            CustomerNotFound / TooManyResults / SaveFailed
        """
        self._require_available()
        repo = self.repository
        customer = self._customer(user_id, customer_id)
        visit = repo.get_visit(user_id, customer_id, visit_id, now_utc())
        if visit is not None and len(visit.get("results") or []) >= (
            MAX_RESULTS_PER_VISIT
        ):
            raise TooManyResults()

        result_id = secrets.token_urlsafe(RESULT_ID_BYTES)
        ext, content_type = (
            ("png", "image/png") if image_format == "png" else ("jpg", "image/jpeg")
        )
        key = pro_record_photo_key(user_id, customer_id, visit_id, result_id, ext)
        try:
            self.storage.save_pro_record_photo(key, image_bytes, content_type)
        except Exception as e:
            logger.error(
                f"❌ 상담 사진 저장 실패: user_id={mask_user_id(user_id)}, "
                f"{type(e).__name__}"
            )
            raise SaveFailed() from e

        result = {"result_id": result_id, "style": style, "photo_key": key}
        # 기록 쓰기가 실패하면 기록에 연결되지 않은 사진은 남기지 않는다
        try:
            saved = self._write_result(
                user_id, customer_id, visit_id, customer, visit, result, gender
            )
        except (TooManyResults, CustomerNotFound, RecordsTableMissing):
            self.storage.delete_object_quietly(key)
            raise
        except Exception as e:
            self.storage.delete_object_quietly(key)
            logger.error(
                f"❌ 상담 기록 저장 실패: user_id={mask_user_id(user_id)}, "
                f"{type(e).__name__}"
            )
            raise SaveFailed() from e

        log_structured(
            "pro_visit_result_saved",
            {
                "user_id": mask_user_id(user_id),
                "result_count": len(saved["results"]),
            },
        )
        return self._public_visit(saved)

    def _write_result(
        self,
        user_id: str,
        customer_id: str,
        visit_id: str,
        customer: Dict[str, Any],
        visit: Optional[Dict[str, Any]],
        result: Dict[str, str],
        gender: str,
    ) -> Dict[str, Any]:
        repo = self.repository
        for attempt in range(_ADD_RESULT_ATTEMPTS):
            if attempt:
                # 다른 요청이 그 사이 상담을 만들었거나 결과를 붙였다 - 다시 읽는다
                now = now_utc()
                customer = repo.get_customer(user_id, customer_id, now)
                if customer is None:
                    raise CustomerNotFound()
                visit = repo.get_visit(user_id, customer_id, visit_id, now)
            if visit is not None and len(visit.get("results") or []) >= (
                MAX_RESULTS_PER_VISIT
            ):
                raise TooManyResults()
            try:
                return repo.add_visit_result(
                    user_id,
                    customer,
                    visit_id,
                    visit,
                    result,
                    gender,
                    now_utc(),
                )
            except VisitChangedError:
                continue
            except CustomerGoneError as e:
                raise CustomerNotFound() from e
        raise RuntimeError("상담 기록이 계속 바뀌어 저장하지 못했습니다")

    def choose_style(
        self, user_id: str, customer_id: str, visit_id: str, style: Any
    ) -> Dict[str, Any]:
        """고른 스타일 저장 (그 상담의 결과 스타일 중 하나여야 한다)

        Raises:
            CustomerNotFound / VisitNotFound / InvalidStyle
        """
        self._require_available()
        self._customer(user_id, customer_id)
        if not is_valid_id(visit_id):
            raise VisitNotFound()
        visit = self.repository.get_visit(user_id, customer_id, visit_id, now_utc())
        if visit is None:
            raise VisitNotFound()
        styles = {r.get("style") for r in visit.get("results") or []}
        if not isinstance(style, str) or style not in styles:
            raise InvalidStyle()
        try:
            saved = self.repository.set_chosen_style(user_id, visit, style)
        except VisitChangedError as e:
            raise VisitNotFound() from e
        return self._public_visit(saved)

    # ========== 회원 탈퇴 ==========

    def delete_all_records(self, user_id: str) -> int:
        """디자이너의 손님 기록 전부 삭제 (사진 → 기록 순)

        테이블이 설정되지 않았거나 아직 없으면 지울 것이 없으므로 0 이다.

        Returns:
            지운 손님 수

        Raises:
            사진·기록 삭제 실패는 그대로 전파 (호출자가 탈퇴를 멈춰야 한다)
        """
        if not self.repository.enabled:
            return 0
        self.storage.delete_pro_records(pro_records_prefix(user_id))
        try:
            return self.repository.delete_all(user_id)
        except RecordsTableMissing:
            return 0


# Singleton
_pro_records_service: Optional[ProRecordsService] = None


def get_pro_records_service() -> ProRecordsService:
    global _pro_records_service
    if _pro_records_service is None:
        _pro_records_service = ProRecordsService()
    return _pro_records_service
