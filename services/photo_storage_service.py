"""사진 저장 서비스 (S3)

버킷 구조 (PHOTO_S3_BUCKET):
- cache/{cache_key}.{fmt}      합성 결과 캐시 (같은 사진+스타일 재요청 시 Gemini 재호출 방지)
- results/{user_id}/{id}.{fmt} 회원별 합성 결과 (마이페이지 재열람용)
- pro/results/{user_id}/{ts}_{share_token}.{fmt} Pro 계정(미용실) 합성 결과. 손님 사진이므로
  손님용 결과와 경로를 나누고 짧게(7일) 보관한다 (scripts/apply_pro_results_lifecycle.py).
  Pro 계정은 cache/ 와 originals/ 를 쓰지 않는다 (api/endpoints/synthesis.py).
- pro/results/_share/{share_token}.json 손님에게 QR 로 보내는 짧은 주소(/s/{share_token})의
  연결 정보 {"key", "created_at"}. 같은 pro/results/ 수명주기 규칙으로 7일 뒤 지워진다.
  user_id 는 UUID hex 라 "_share" 와 겹치지 않는다. 토큰을 결과 파일 이름에도 넣어 두어
  회원 탈퇴 때 연결 정보까지 찾아 지울 수 있다.
- originals/{user_id}/{id}.jpg 원본 사진 (training_consent=True인 회원만, AI 학습용)
- reports/{YYYY-MM-DD}/{id}.json AI 생성 이미지 신고 (결과 이미지 자체는 저장하지 않음)
- body/originals/{subject}/{id}.{ext} 체형 코디 전신 사진 (분석한 사진 전부, 1년 보관).
  얼굴(헤어) 사진과 경로·보관 기간을 분리한다.

PHOTO_S3_BUCKET 미설정 시 모든 기능이 조용히 비활성화됨 (합성 자체는 정상 동작).
"""

import base64
import hashlib
import json
import os
import re
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote, unquote

try:
    import boto3
    from botocore.exceptions import ClientError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from config.settings import settings
from core.logging import logger


# 체형 사진 경로 (수명주기 규칙도 이 접두사로 건다)
BODY_ORIGINALS_PREFIX = "body/originals/"

# 회원 결과 경로. Pro 경로는 손님용 "results/" 로 시작하지 않으므로
# 손님용 수명주기 규칙(results/ 180일)에 걸리지 않는다.
RESULTS_PREFIX = "results/"
PRO_RESULTS_PREFIX = "pro/results/"
# Pro 결과 공유 주소의 연결 정보 (pro/results/ 수명주기 규칙에 함께 걸린다)
PRO_SHARE_PREFIX = f"{PRO_RESULTS_PREFIX}_share/"

# 공유 토큰: secrets.token_urlsafe(16) = 128비트 난수, URL-safe base64 22자
SHARE_TOKEN_BYTES = 16
SHARE_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_-]{22}")

# S3 delete_objects 한 번에 지울 수 있는 최대 개수
_DELETE_BATCH = 1000


def user_results_prefix(user_id: str, pro: bool = False) -> str:
    """회원 결과가 저장되는 S3 접두사"""
    return f"{PRO_RESULTS_PREFIX if pro else RESULTS_PREFIX}{user_id}/"


def share_pointer_key(token: str) -> str:
    return f"{PRO_SHARE_PREFIX}{token}.json"


def share_token_of_key(key: str) -> Optional[str]:
    """Pro 결과 파일 이름({ts}_{token}.{fmt})에서 공유 토큰을 꺼낸다 (없으면 None)"""
    name = key.rsplit("/", 1)[-1]
    _, sep, rest = name.partition("_")
    token = rest.rsplit(".", 1)[0] if sep else ""
    return token if SHARE_TOKEN_PATTERN.fullmatch(token) else None


class ShareLinkExpired(Exception):
    """공유 주소의 보관 기간이 지났다"""


PNG_SIGNATURE = bytes([0x89]) + b"PNG"


def _image_type(data: bytes) -> Tuple[str, str]:
    """매직 바이트로 확장자/ContentType 판별 (업로드 검증을 통과한 jpg/png/webp)"""
    if data.startswith(PNG_SIGNATURE):
        return "png", "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp", "image/webp"
    return "jpg", "image/jpeg"


class PhotoStorageService:
    """S3 기반 사진 저장/결과 캐싱"""

    def __init__(self):
        self._client = None

    @property
    def enabled(self) -> bool:
        return bool(self.bucket) and BOTO3_AVAILABLE

    @property
    def bucket(self) -> str:
        return os.getenv("PHOTO_S3_BUCKET", settings.PHOTO_S3_BUCKET)

    @property
    def client(self):
        if self._client is None:
            if not BOTO3_AVAILABLE:
                raise RuntimeError("boto3 is not installed")
            aws_region = os.getenv("AWS_REGION", settings.AWS_REGION)
            # 리전 엔드포인트 고정 필수: 글로벌 엔드포인트(s3.amazonaws.com)로
            # presigned URL이 생성되면 SignatureDoesNotMatch(403)로 이미지 로드 실패
            config = Config(
                connect_timeout=5,
                read_timeout=15,
                retries={"max_attempts": 3},
                signature_version="s3v4",
                s3={"addressing_style": "virtual"},
            )
            self._client = boto3.client(
                "s3",
                region_name=aws_region,
                endpoint_url=f"https://s3.{aws_region}.amazonaws.com",
                config=config,
            )
        return self._client

    # ========== 결과 캐싱 (비용 절감 핵심) ==========

    @staticmethod
    def build_cache_key(*parts: bytes) -> str:
        """이미지 바이트 + 스타일 파라미터로 캐시 키 생성 (SHA-256)"""
        h = hashlib.sha256()
        for part in parts:
            h.update(part)
            h.update(b"|")
        return h.hexdigest()

    def get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """캐시된 합성 결과 조회. 없으면 None"""
        if not self.enabled:
            return None

        key = f"cache/{cache_key}"
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
            image_bytes = response["Body"].read()
            image_format = response.get("Metadata", {}).get("image-format", "png")
            logger.info(f"💰 합성 캐시 히트: {cache_key[:12]}... (Gemini 호출 생략)")
            return {
                "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
                "image_format": image_format,
            }
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            logger.warning(f"캐시 조회 실패 (무시): {e.response['Error']['Message']}")
            return None
        except Exception as e:
            logger.warning(f"캐시 조회 실패 (무시): {str(e)}")
            return None

    def save_cached_result(
        self, cache_key: str, image_base64: str, image_format: str = "png"
    ) -> None:
        """합성 결과를 캐시에 저장 (실패해도 응답에 영향 없음)"""
        if not self.enabled:
            return

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=f"cache/{cache_key}",
                Body=base64.b64decode(image_base64),
                ContentType=f"image/{image_format}",
                Metadata={"image-format": image_format},
            )
        except Exception as e:
            logger.warning(f"캐시 저장 실패 (무시): {str(e)}")

    # ========== 회원 결과 저장 ==========

    def save_user_result(
        self,
        user_id: str,
        image_base64: str,
        image_format: str = "png",
        hairstyle_name: Optional[str] = None,
        pro: bool = False,
    ) -> Optional[str]:
        """
        회원의 합성 결과 저장 후 presigned URL 반환

        pro: Pro 계정이면 pro/results/ 아래에 저장한다 (공유 주소도 만든다 - save_pro_result)

        Returns:
            presigned URL (실패 시 None - 합성 응답은 정상 진행)
        """
        if pro:
            saved = self.save_pro_result(
                user_id, image_base64, image_format, hairstyle_name
            )
            return saved["result_url"] if saved else None
        if not self.enabled:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = (
            f"{user_results_prefix(user_id, pro)}{ts}_{uuid.uuid4().hex[:8]}"
            f".{image_format}"
        )

        metadata = {}
        if hairstyle_name:
            # S3 메타데이터는 ASCII만 허용 - 한글 스타일명은 URL 인코딩
            metadata["hairstyle"] = quote(hairstyle_name)

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=base64.b64decode(image_base64),
                ContentType=f"image/{image_format}",
                Metadata=metadata,
            )
            return self.presigned_url(key)
        except Exception as e:
            logger.warning(f"결과 저장 실패 (무시): {str(e)}")
            return None

    def save_pro_result(
        self,
        user_id: str,
        image_base64: str,
        image_format: str = "png",
        hairstyle_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Pro 계정 결과 저장 + 손님에게 보낼 공유 토큰 발급

        결과 파일 이름에 토큰을 넣고(pro/results/{user_id}/{ts}_{token}.{fmt}),
        토큰 → 파일 연결 정보를 pro/results/_share/{token}.json 에 쓴다.
        연결 정보 저장이 실패하면 share_token 만 None 이다 (결과 자체는 저장됨).

        Returns:
            {"result_url", "share_token", "created_at"(datetime)} (결과 저장 실패 시 None)
        """
        if not self.enabled:
            return None

        token = secrets.token_urlsafe(SHARE_TOKEN_BYTES)
        now = datetime.now(timezone.utc)
        key = (
            f"{user_results_prefix(user_id, True)}"
            f"{now.strftime('%Y%m%dT%H%M%S')}_{token}.{image_format}"
        )
        metadata = {}
        if hairstyle_name:
            metadata["hairstyle"] = quote(hairstyle_name)

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=base64.b64decode(image_base64),
                ContentType=f"image/{image_format}",
                Metadata=metadata,
            )
        except Exception as e:
            logger.warning(f"Pro 결과 저장 실패 (무시): {str(e)}")
            return None

        share_token: Optional[str] = token
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=share_pointer_key(token),
                Body=json.dumps({"key": key, "created_at": now.isoformat()}).encode(
                    "utf-8"
                ),
                ContentType="application/json",
            )
        except Exception as e:
            logger.warning(f"공유 주소 저장 실패 (결과는 저장됨): {str(e)}")
            share_token = None

        return {
            "result_url": self.presigned_url(key),
            "share_token": share_token,
            "created_at": now,
        }

    def resolve_share_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        공유 토큰 → 지금 열 수 있는 이미지 주소

        Returns:
            {"image_url", "expires_at"(datetime)} / 없는 토큰·지워진 결과는 None

        Raises:
            ShareLinkExpired: 보관 기간(PRO_RESULT_RETENTION_DAYS)이 지남
            그 밖의 S3 오류는 그대로 전파 (호출자가 503 처리)
        """
        if not self.enabled or not SHARE_TOKEN_PATTERN.fullmatch(token or ""):
            return None
        try:
            response = self.client.get_object(
                Bucket=self.bucket, Key=share_pointer_key(token)
            )
            pointer = json.loads(response["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

        key = pointer.get("key")
        # 다른 경로를 가리키는 연결 정보는 믿지 않는다
        if not isinstance(key, str) or not key.startswith(PRO_RESULTS_PREFIX):
            return None
        created_at = datetime.fromisoformat(pointer["created_at"])
        expires_at = created_at + timedelta(days=settings.PRO_RESULT_RETENTION_DAYS)
        # 수명주기 규칙은 자정 단위로 늦게 지우므로 기간은 서버가 직접 지킨다
        if datetime.now(timezone.utc) >= expires_at:
            raise ShareLinkExpired()

        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404", "NotFound"):
                return None  # 탈퇴 등으로 결과가 지워짐
            raise

        image_url = self.presigned_url(key, settings.PRO_SHARE_IMAGE_URL_SECONDS)
        if image_url is None:
            raise RuntimeError("presigned URL 생성 실패")
        return {"image_url": image_url, "expires_at": expires_at}

    def delete_user_results(self, user_id: str, pro: bool) -> int:
        """
        회원 결과 전부 삭제 (회원 탈퇴용). Pro 결과면 공유 주소 연결 정보도 지운다.

        Returns:
            지운 결과 사진 수

        Raises:
            S3 조회·삭제 실패 시 예외 (호출자가 탈퇴를 중단해야 한다)
        """
        if not self.enabled:
            return 0

        prefix = user_results_prefix(user_id, pro)
        result_keys: List[str] = []
        list_kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        while True:
            response = self.client.list_objects_v2(**list_kwargs)
            result_keys.extend(obj["Key"] for obj in response.get("Contents", []))
            if not response.get("IsTruncated"):
                break
            list_kwargs["ContinuationToken"] = response["NextContinuationToken"]

        keys = list(result_keys)
        if pro:
            for key in result_keys:
                token = share_token_of_key(key)
                if token:
                    keys.append(share_pointer_key(token))

        for start in range(0, len(keys), _DELETE_BATCH):
            batch = keys[start : start + _DELETE_BATCH]
            response = self.client.delete_objects(
                Bucket=self.bucket,
                Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
            )
            errors = response.get("Errors") or []
            if errors:
                raise RuntimeError(f"결과 삭제 일부 실패: {len(errors)}건")

        return len(result_keys)

    def list_user_results(
        self,
        user_id: str,
        limit: int = 20,
        continuation_token: Optional[str] = None,
        pro: bool = False,
    ) -> Dict[str, Any]:
        """
        회원의 합성 결과 목록 (최신순, presigned URL 포함)

        키 형식이 results/{user_id}/{UTC타임스탬프}_{난수}.{fmt} 이므로
        키 내림차순 정렬 = 최신순. S3 list는 오름차순만 지원하므로
        prefix 전체 키를 모은 뒤 정렬한다 (수명주기 180일이라 개수 제한적).

        Args:
            continuation_token: 이전 페이지 마지막 항목의 key (이후 과거분 조회)
            pro: Pro 계정이면 pro/results/ 아래만 조회한다

        Returns:
            {"items": [{"key", "url", "hairstyle", "created_at"}], "next_token": str|None}

        Raises:
            S3 조회 실패 시 예외 전파 (호출자가 500 처리)
        """
        if not self.enabled:
            return {"items": [], "next_token": None}

        prefix = user_results_prefix(user_id, pro)
        keys: List[Tuple[str, Any]] = []  # (key, last_modified)
        list_kwargs: Dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
        while True:
            response = self.client.list_objects_v2(**list_kwargs)
            for obj in response.get("Contents", []):
                keys.append((obj["Key"], obj.get("LastModified")))
            if not response.get("IsTruncated"):
                break
            list_kwargs["ContinuationToken"] = response["NextContinuationToken"]

        keys.sort(key=lambda entry: entry[0], reverse=True)  # 최신순
        if continuation_token:
            keys = [entry for entry in keys if entry[0] < continuation_token]

        page = keys[:limit]
        next_token = page[-1][0] if len(keys) > limit else None

        items = []
        for key, last_modified in page:
            hairstyle = None
            try:
                head = self.client.head_object(Bucket=self.bucket, Key=key)
                raw = head.get("Metadata", {}).get("hairstyle")
                if raw:
                    # 저장 시 URL 인코딩된 한글 스타일명 복원
                    hairstyle = unquote(raw)
            except Exception as e:
                logger.warning(f"결과 메타데이터 조회 실패 (무시): {str(e)}")

            items.append(
                {
                    "key": key,
                    "url": self.presigned_url(key),
                    "hairstyle": hairstyle,
                    "created_at": (
                        last_modified.isoformat() if last_modified else None
                    ),
                }
            )

        return {"items": items, "next_token": next_token}

    # ========== 원본 사진 저장 (동의한 회원만) ==========

    def save_original_photo(
        self, user_id: str, image_bytes: bytes, content_type: str = "image/jpeg"
    ) -> Optional[str]:
        """
        학습 활용에 동의(training_consent=True)한 회원의 원본 사진 저장

        호출자가 동의 여부를 확인한 뒤 호출해야 한다.

        Returns:
            저장된 S3 key (실패 시 None)
        """
        if not self.enabled:
            return None

        ext = content_type.split("/")[-1] if "/" in content_type else "jpg"
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"originals/{user_id}/{ts}_{uuid.uuid4().hex[:8]}.{ext}"

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType=content_type,
                Metadata={"training-consent": "true"},
            )
            logger.info(f"📸 원본 사진 저장 (학습 동의): user_id={user_id}")
            return key
        except Exception as e:
            logger.warning(f"원본 저장 실패 (무시): {str(e)}")
            return None

    # ========== AI 생성 이미지 신고 ==========

    def save_report(self, report: Dict[str, Any]) -> Optional[str]:
        """신고 1건을 JSON 으로 저장. 저장된 S3 key (비활성/실패 시 None)"""
        if not self.enabled:
            return None

        now = datetime.now(timezone.utc)
        key = (
            f"reports/{now.strftime('%Y-%m-%d')}/"
            f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}.json"
        )
        body = {**report, "created_at": now.isoformat()}
        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
            )
            return key
        except Exception as e:
            logger.warning(f"신고 저장 실패: {str(e)}")
            return None

    # ========== 체형 코디 전신 사진 (1년 보관) ==========

    def save_body_photo(
        self, subject: str, image_bytes: bytes, metadata: Dict[str, str]
    ) -> Optional[str]:
        """
        체형 분석 사진 저장. 헤어 원본(originals/)과 섞이지 않게 body/originals/ 아래에 둔다.

        subject 는 "user-{user_id}" 또는 "device-{device_id}" (경로에 쓸 수 있게 정리된 값).

        Returns:
            저장된 S3 key (비활성/실패 시 None)
        """
        if not self.enabled:
            return None

        ext, content_type = _image_type(image_bytes)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"{BODY_ORIGINALS_PREFIX}{subject}/{ts}_{uuid.uuid4().hex[:8]}.{ext}"

        try:
            self.client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=image_bytes,
                ContentType=content_type,
                Metadata={"source": "body-analysis", **metadata},
            )
            logger.info(f"🧍 체형 사진 저장: {subject}")
            return key
        except Exception as e:
            logger.warning(f"체형 사진 저장 실패 (무시): {str(e)}")
            return None

    def presigned_url(
        self, key: str, expires_in: Optional[int] = None
    ) -> Optional[str]:
        """조회용 presigned URL 생성

        Lambda 에서는 실행 역할의 임시 자격증명으로 서명하므로, 그 자격증명이 먼저
        만료되면 ExpiresIn 보다 일찍 열리지 않는다 (AWS 문서 기준).
        """
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in or settings.PHOTO_URL_EXPIRE_SECONDS,
            )
        except Exception as e:
            logger.warning(f"presigned URL 생성 실패: {str(e)}")
            return None


# Singleton
_photo_storage_service: Optional[PhotoStorageService] = None


def get_photo_storage_service() -> PhotoStorageService:
    global _photo_storage_service
    if _photo_storage_service is None:
        _photo_storage_service = PhotoStorageService()
    return _photo_storage_service
