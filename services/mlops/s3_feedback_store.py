"""
S3 피드백 저장소

피드백 데이터를 S3에 NPZ 포맷으로 저장하여 ML 학습에 사용합니다.
DynamoDB 저장과 병행하여 학습 데이터를 효율적으로 관리합니다.

Architecture:
    s3://hairme-mlops/
    ├── feedback/
    │   ├── pending/           # 학습 대기 중인 피드백
    │   │   ├── 2025-12-02_001.npz
    │   │   └── 2025-12-02_002.npz
    │   ├── processed/         # 학습 완료된 피드백
    │   │   └── batch_2025-12-02.npz
    │   └── metadata.json      # 피드백 카운터 및 상태
    ├── models/
    │   ├── current/           # 현재 운영 모델
    │   │   └── model_v6.pt
    │   └── archive/           # 이전 모델 백업
    │       └── model_v5.pt
    └── training/
        ├── logs/              # 학습 로그
        └── checkpoints/       # 학습 체크포인트

비용:
    - S3 Standard: $0.023/GB/월
    - PUT 요청: $0.005/1000건
    - 예상 월 비용: $0.02 미만 (피드백 1000건 기준)

Author: HairMe ML Team
Date: 2025-12-02
"""

import os
import io
import json
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET_NAME = os.getenv('MLOPS_S3_BUCKET', 'hairme-mlops')
S3_FEEDBACK_PREFIX = 'feedback/pending/'
S3_PROCESSED_PREFIX = 'feedback/processed/'
S3_METADATA_KEY = 'feedback/metadata.json'
S3_MODELS_PREFIX = 'models/'

# 재학습 트리거 임계값
RETRAIN_THRESHOLD = int(os.getenv('MLOPS_RETRAIN_THRESHOLD', '100'))

# 로컬 임베딩 경로 (Lambda TASK_ROOT 기반)
_LAMBDA_TASK_ROOT = os.getenv('LAMBDA_TASK_ROOT', '/var/task')
LOCAL_EMBEDDINGS_PATH = os.getenv(
    'STYLE_EMBEDDINGS_PATH',
    os.path.join(_LAMBDA_TASK_ROOT, 'data_source', 'style_embeddings.npz')
)

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class S3FeedbackStore:
    """
    S3 기반 피드백 저장소

    피드백을 NPZ 포맷으로 S3에 저장하고, 학습 데이터로 변환합니다.
    """

    def __init__(
        self,
        bucket_name: str = S3_BUCKET_NAME,
        region: str = 'ap-northeast-2'
    ):
        """
        초기화

        Args:
            bucket_name: S3 버킷 이름
            region: AWS 리전
        """
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = None
        self.enabled = False

        # 스타일 임베딩 로드
        self.style_embeddings = None
        self.style_to_idx = {}

        self._init_s3_client()
        self._load_style_embeddings()

    def _init_s3_client(self):
        """S3 클라이언트 초기화"""
        if not BOTO3_AVAILABLE:
            logger.warning("⚠️ boto3 not installed - S3 피드백 저장 비활성화")
            return

        # MLOps 활성화 여부 확인
        mlops_enabled = os.getenv('MLOPS_ENABLED', 'false').lower() == 'true'
        if not mlops_enabled:
            logger.info("ℹ️ MLOps 비활성화 (MLOPS_ENABLED=false)")
            return

        try:
            self.s3_client = boto3.client('s3', region_name=self.region)

            # 버킷 존재 확인
            self.s3_client.head_bucket(Bucket=self.bucket_name)

            self.enabled = True
            logger.info(f"✅ S3 피드백 저장소 초기화 완료: {self.bucket_name}")

        except NoCredentialsError:
            logger.warning("⚠️ AWS 자격 증명 없음 - S3 피드백 저장 비활성화")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.warning(f"⚠️ S3 버킷 없음: {self.bucket_name} - 자동 생성 시도")
                self._create_bucket()
            else:
                logger.error(f"❌ S3 초기화 실패: {e}")
        except Exception as e:
            logger.error(f"❌ S3 초기화 실패: {e}")

    def _create_bucket(self):
        """S3 버킷 생성"""
        try:
            if self.region == 'us-east-1':
                self.s3_client.create_bucket(Bucket=self.bucket_name)
            else:
                self.s3_client.create_bucket(
                    Bucket=self.bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )

            # 버킷 버저닝 활성화 (모델 롤백용)
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )

            self.enabled = True
            logger.info(f"✅ S3 버킷 생성 완료: {self.bucket_name}")

            # 초기 메타데이터 생성
            self._init_metadata()

        except Exception as e:
            logger.error(f"❌ S3 버킷 생성 실패: {e}")

    def _init_metadata(self):
        """초기 메타데이터 파일 생성"""
        metadata = {
            "total_feedback_count": 0,
            "pending_count": 0,
            "last_training_at": None,
            "last_feedback_at": None,
            "model_version": "v6",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        self._save_metadata(metadata)

    def _load_style_embeddings(self):
        """헤어스타일 임베딩 로드"""
        try:
            # Lambda 환경에서는 LAMBDA_TASK_ROOT 기반 절대 경로 사용
            paths_to_try = [
                LOCAL_EMBEDDINGS_PATH,
                os.path.join(_LAMBDA_TASK_ROOT, 'data_source', 'style_embeddings.npz'),
                '/tmp/style_embeddings.npz',
            ]

            logger.info(f"🔍 스타일 임베딩 검색 경로: {paths_to_try}")

            for path in paths_to_try:
                if os.path.exists(path):
                    data = np.load(path, allow_pickle=False)
                    styles = data['styles'].tolist()
                    self.style_embeddings = data['embeddings']
                    self.style_to_idx = {style: idx for idx, style in enumerate(styles)}
                    logger.info(f"✅ 스타일 임베딩 로드 완료: {len(styles)}개 ({path})")
                    return

            logger.warning(f"⚠️ 스타일 임베딩 파일을 찾을 수 없음: {paths_to_try}")

        except Exception as e:
            logger.error(f"❌ 스타일 임베딩 로드 실패: {e}")

    def _get_metadata(self) -> Dict[str, Any]:
        """메타데이터 조회"""
        if not self.enabled:
            return {}

        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=S3_METADATA_KEY
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                self._init_metadata()
                return self._get_metadata()
            raise

    def _save_metadata(self, metadata: Dict[str, Any]):
        """메타데이터 저장"""
        if not self.enabled:
            return

        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=S3_METADATA_KEY,
            Body=json.dumps(metadata, ensure_ascii=False, indent=2),
            ContentType='application/json'
        )

    def get_style_embedding(self, hairstyle_id) -> Optional[np.ndarray]:
        """
        헤어스타일 ID로 임베딩 가져오기

        Args:
            hairstyle_id: 헤어스타일 ID (0-based index) - int, Decimal, str 모두 허용

        Returns:
            임베딩 벡터 (384차원) 또는 None
        """
        if self.style_embeddings is None:
            logger.warning("⚠️ 스타일 임베딩이 로드되지 않음")
            return None

        # DynamoDB Decimal 또는 문자열을 정수로 변환
        try:
            hairstyle_id = int(hairstyle_id)
        except (ValueError, TypeError) as e:
            logger.warning(f"⚠️ hairstyle_id 변환 실패: {hairstyle_id} ({type(hairstyle_id)})")
            return None

        if 0 <= hairstyle_id < len(self.style_embeddings):
            return self.style_embeddings[hairstyle_id]

        logger.warning(f"⚠️ 잘못된 hairstyle_id: {hairstyle_id} (범위: 0-{len(self.style_embeddings)-1})")
        return None

    def save_feedback(
        self,
        analysis_id: str,
        face_shape: str,
        skin_tone: str,
        hairstyle_id: int,
        feedback: str,  # 'good' or 'bad'
        face_features: Optional[List[float]] = None,
        skin_features: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        피드백을 S3에 저장

        Args:
            analysis_id: 분석 ID (UUID)
            face_shape: 얼굴형
            skin_tone: 피부톤
            hairstyle_id: 헤어스타일 ID
            feedback: 피드백 ('good' or 'bad')
            face_features: MediaPipe 얼굴 측정값 (6차원)
            skin_features: MediaPipe 피부 측정값 (2차원)

        Returns:
            {
                "success": bool,
                "pending_count": int,
                "should_trigger_training": bool
            }
        """
        if not self.enabled:
            logger.debug("S3 피드백 저장 비활성화 상태")
            return {
                "success": False,
                "pending_count": 0,
                "should_trigger_training": False
            }

        try:
            # 1. Ground truth 결정
            ground_truth = 90.0 if feedback == 'good' else 10.0

            # 2. 스타일 임베딩 가져오기
            style_embedding = self.get_style_embedding(hairstyle_id)
            if style_embedding is None:
                logger.warning(f"⚠️ 스타일 임베딩 없음 (ID: {hairstyle_id}) - 피드백 저장 건너뜀")
                return {
                    "success": False,
                    "pending_count": 0,
                    "should_trigger_training": False
                }

            # 3. Feature 벡터 구성
            if face_features is not None and skin_features is not None:
                # 실제 측정값 사용 (6 + 2 = 8차원)
                face_vec = np.array(face_features, dtype=np.float32)
                skin_vec = np.array(skin_features, dtype=np.float32)
            else:
                # 레거시: 라벨 기반 인코딩 (4 + 4 = 8차원)
                face_vec = self._encode_face_shape(face_shape)
                skin_vec = self._encode_skin_tone(skin_tone)

            # 4. NPZ 데이터 준비
            timestamp = datetime.now(timezone.utc)
            filename = f"{timestamp.strftime('%Y-%m-%d')}_{analysis_id[:8]}.npz"

            # 학습에 필요한 모든 데이터 저장
            npz_data = {
                'face_features': face_vec,
                'skin_features': skin_vec,
                'style_embedding': style_embedding.astype(np.float32),
                'ground_truth': np.array([ground_truth], dtype=np.float32),
                'metadata': np.array([json.dumps({
                    'analysis_id': analysis_id,
                    'face_shape': face_shape,
                    'skin_tone': skin_tone,
                    'hairstyle_id': hairstyle_id,
                    'feedback': feedback,
                    'timestamp': timestamp.isoformat()
                })], dtype=str)
            }

            # 5. S3에 업로드
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **npz_data)
            buffer.seek(0)

            s3_key = f"{S3_FEEDBACK_PREFIX}{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue(),
                ContentType='application/octet-stream'
            )

            # 6. 메타데이터 업데이트
            metadata = self._get_metadata()
            metadata['total_feedback_count'] = metadata.get('total_feedback_count', 0) + 1
            metadata['pending_count'] = metadata.get('pending_count', 0) + 1
            metadata['last_feedback_at'] = timestamp.isoformat()
            self._save_metadata(metadata)

            pending_count = metadata['pending_count']
            should_trigger = pending_count >= RETRAIN_THRESHOLD

            logger.info(
                f"✅ S3 피드백 저장 완료: {s3_key} | "
                f"pending: {pending_count}/{RETRAIN_THRESHOLD}"
            )

            return {
                "success": True,
                "pending_count": pending_count,
                "should_trigger_training": should_trigger
            }

        except Exception as e:
            logger.error(f"❌ S3 피드백 저장 실패: {e}")
            return {
                "success": False,
                "pending_count": 0,
                "should_trigger_training": False
            }

    def _encode_face_shape(self, face_shape: str) -> np.ndarray:
        """얼굴형 라벨 인코딩 (레거시 지원)"""
        FACE_SHAPES = ["각진형", "둥근형", "긴형", "계란형"]
        vec = np.zeros(6, dtype=np.float32)

        if face_shape == "하트형":
            face_shape = "계란형"

        if face_shape in FACE_SHAPES:
            idx = FACE_SHAPES.index(face_shape)
            vec[idx] = 1.0
        else:
            vec[3] = 1.0  # 기본값: 계란형

        return vec

    def _encode_skin_tone(self, skin_tone: str) -> np.ndarray:
        """피부톤 라벨 인코딩 (레거시 지원)"""
        SKIN_TONES = ["겨울쿨", "가을웜", "봄웜", "여름쿨"]
        vec = np.zeros(2, dtype=np.float32)

        # 웜톤/쿨톤으로 단순화
        if skin_tone in ["봄웜", "가을웜"]:
            vec[0] = 1.0  # 웜톤
        elif skin_tone in ["여름쿨", "겨울쿨"]:
            vec[1] = 1.0  # 쿨톤
        else:
            vec[0] = 1.0  # 기본값: 웜톤

        return vec

    def get_pending_feedbacks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        대기 중인 피드백 데이터 로드 (학습용)

        Returns:
            (face_features, skin_features, style_embeddings, ground_truths), count
            각각 (N, dim) 형태의 numpy 배열
        """
        if not self.enabled:
            return None, None, None, None, 0

        try:
            # pending 폴더의 모든 파일 목록
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=S3_FEEDBACK_PREFIX
            )

            if 'Contents' not in response:
                return None, None, None, None, 0

            face_list = []
            skin_list = []
            style_list = []
            gt_list = []

            for obj in response['Contents']:
                key = obj['Key']
                if not key.endswith('.npz'):
                    continue

                # NPZ 파일 다운로드 및 파싱
                obj_response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=key
                )

                buffer = io.BytesIO(obj_response['Body'].read())
                data = np.load(buffer, allow_pickle=True)

                face_list.append(data['face_features'])
                skin_list.append(data['skin_features'])
                style_list.append(data['style_embedding'])
                gt_list.append(data['ground_truth'])

            if not face_list:
                return None, None, None, None, 0

            # 배열로 변환
            face_features = np.stack(face_list)
            skin_features = np.stack(skin_list)
            style_embeddings = np.stack(style_list)
            ground_truths = np.concatenate(gt_list)

            logger.info(f"✅ {len(face_list)}개 피드백 데이터 로드 완료")

            return face_features, skin_features, style_embeddings, ground_truths, len(face_list)

        except Exception as e:
            logger.error(f"❌ 피드백 데이터 로드 실패: {e}")
            return None, None, None, None, 0

    def mark_feedbacks_processed(self, batch_name: Optional[str] = None):
        """
        pending 피드백을 processed로 이동

        Args:
            batch_name: 배치 이름 (없으면 자동 생성)
        """
        if not self.enabled:
            return

        try:
            timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')
            batch_name = batch_name or f"batch_{timestamp}"

            # pending 파일 목록
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=S3_FEEDBACK_PREFIX
            )

            if 'Contents' not in response:
                return

            moved_count = 0
            for obj in response['Contents']:
                old_key = obj['Key']
                if not old_key.endswith('.npz'):
                    continue

                # processed 폴더로 이동
                filename = old_key.split('/')[-1]
                new_key = f"{S3_PROCESSED_PREFIX}{batch_name}/{filename}"

                # Copy then delete
                self.s3_client.copy_object(
                    Bucket=self.bucket_name,
                    CopySource={'Bucket': self.bucket_name, 'Key': old_key},
                    Key=new_key
                )
                self.s3_client.delete_object(
                    Bucket=self.bucket_name,
                    Key=old_key
                )
                moved_count += 1

            # 메타데이터 업데이트
            metadata = self._get_metadata()
            metadata['pending_count'] = 0
            metadata['last_training_at'] = datetime.now(timezone.utc).isoformat()
            self._save_metadata(metadata)

            logger.info(f"✅ {moved_count}개 피드백을 processed로 이동: {batch_name}")

        except Exception as e:
            logger.error(f"❌ 피드백 이동 실패: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """피드백 통계 조회"""
        if not self.enabled:
            return {
                "enabled": False,
                "total_feedback_count": 0,
                "pending_count": 0
            }

        try:
            metadata = self._get_metadata()
            return {
                "enabled": True,
                "bucket": self.bucket_name,
                "total_feedback_count": metadata.get('total_feedback_count', 0),
                "pending_count": metadata.get('pending_count', 0),
                "retrain_threshold": RETRAIN_THRESHOLD,
                "last_feedback_at": metadata.get('last_feedback_at'),
                "last_training_at": metadata.get('last_training_at'),
                "model_version": metadata.get('model_version', 'unknown')
            }
        except Exception as e:
            logger.error(f"❌ 통계 조회 실패: {e}")
            return {"enabled": True, "error": str(e)}


# ========== 싱글톤 인스턴스 ==========
_store_instance = None


def get_s3_feedback_store() -> S3FeedbackStore:
    """
    S3 피드백 저장소 싱글톤 인스턴스

    Returns:
        S3FeedbackStore 인스턴스
    """
    global _store_instance

    if _store_instance is None:
        logger.info("🔧 S3 피드백 저장소 초기화 중...")
        _store_instance = S3FeedbackStore()

    return _store_instance
