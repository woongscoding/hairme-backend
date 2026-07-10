"""사진 저장 서비스 (S3)

버킷 구조 (PHOTO_S3_BUCKET):
- cache/{cache_key}.{fmt}      합성 결과 캐시 (같은 사진+스타일 재요청 시 Gemini 재호출 방지)
- results/{user_id}/{id}.{fmt} 회원별 합성 결과 (마이페이지 재열람용)
- originals/{user_id}/{id}.jpg 원본 사진 (training_consent=True인 회원만, AI 학습용)

PHOTO_S3_BUCKET 미설정 시 모든 기능이 조용히 비활성화됨 (합성 자체는 정상 동작).
"""

import base64
import hashlib
import os
import uuid
from datetime import datetime, timezone
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
            config = Config(
                connect_timeout=5, read_timeout=15, retries={"max_attempts": 3}
            )
            self._client = boto3.client("s3", region_name=aws_region, config=config)
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
    ) -> Optional[str]:
        """
        회원의 합성 결과 저장 후 presigned URL 반환

        Returns:
            presigned URL (실패 시 None - 합성 응답은 정상 진행)
        """
        if not self.enabled:
            return None

        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        key = f"results/{user_id}/{ts}_{uuid.uuid4().hex[:8]}.{image_format}"

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

    def list_user_results(
        self,
        user_id: str,
        limit: int = 20,
        continuation_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        회원의 합성 결과 목록 (최신순, presigned URL 포함)

        키 형식이 results/{user_id}/{UTC타임스탬프}_{난수}.{fmt} 이므로
        키 내림차순 정렬 = 최신순. S3 list는 오름차순만 지원하므로
        prefix 전체 키를 모은 뒤 정렬한다 (수명주기 180일이라 개수 제한적).

        Args:
            continuation_token: 이전 페이지 마지막 항목의 key (이후 과거분 조회)

        Returns:
            {"items": [{"key", "url", "hairstyle", "created_at"}], "next_token": str|None}

        Raises:
            S3 조회 실패 시 예외 전파 (호출자가 500 처리)
        """
        if not self.enabled:
            return {"items": [], "next_token": None}

        prefix = f"results/{user_id}/"
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

    def presigned_url(self, key: str) -> Optional[str]:
        """조회용 presigned URL 생성"""
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=settings.PHOTO_URL_EXPIRE_SECONDS,
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
