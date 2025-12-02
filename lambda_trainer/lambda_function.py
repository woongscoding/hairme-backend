"""
HairMe ML Trainer Lambda

EventBridge 또는 수동 트리거로 실행됩니다.
S3에서 피드백 데이터를 가져와 모델을 재학습합니다.

Note: 실제 학습은 Lambda 메모리/시간 제한으로 인해
      간단한 fine-tuning만 수행하거나,
      EC2 Spot Instance를 시작하는 역할을 합니다.
"""

import json
import os
import logging
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.getenv('MLOPS_S3_BUCKET', 'hairme-mlops')
MIN_SAMPLES = int(os.getenv('MLOPS_MIN_SAMPLES', '50'))


def get_pending_count():
    """S3에서 pending 피드백 수 확인"""
    import boto3

    s3 = boto3.client('s3')

    try:
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json'
        )
        metadata = json.loads(response['Body'].read().decode('utf-8'))
        return metadata.get('pending_count', 0)
    except Exception as e:
        logger.error(f"Failed to get metadata: {e}")
        return 0


def update_metadata(pending_count: int, training_triggered: bool = False):
    """메타데이터 업데이트"""
    import boto3

    s3 = boto3.client('s3')

    try:
        # 기존 메타데이터 로드
        response = s3.get_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json'
        )
        metadata = json.loads(response['Body'].read().decode('utf-8'))

        # 업데이트
        if training_triggered:
            metadata['last_training_at'] = datetime.now(timezone.utc).isoformat()
            metadata['pending_count'] = 0
        else:
            metadata['pending_count'] = pending_count

        # 저장
        s3.put_object(
            Bucket=S3_BUCKET,
            Key='feedback/metadata.json',
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )

    except Exception as e:
        logger.error(f"Failed to update metadata: {e}")


def lambda_handler(event, context):
    """
    Lambda 핸들러

    Args:
        event: {
            "trigger_type": "scheduled" | "data_threshold" | "manual",
            "metadata": {...}
        }
    """
    logger.info(f"Received event: {json.dumps(event)}")

    trigger_type = event.get('trigger_type', 'unknown')
    timestamp = datetime.now(timezone.utc).isoformat()

    # Pending 피드백 수 확인
    pending_count = get_pending_count()
    logger.info(f"Pending feedback count: {pending_count}")

    # 최소 샘플 수 확인
    if pending_count < MIN_SAMPLES:
        message = f"Insufficient data: {pending_count}/{MIN_SAMPLES} samples"
        logger.info(message)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': False,
                'message': message,
                'trigger_type': trigger_type,
                'pending_count': pending_count,
                'min_samples': MIN_SAMPLES,
                'timestamp': timestamp
            })
        }

    # TODO: 실제 학습 로직
    # Option 1: Lambda 내에서 간단한 fine-tuning (메모리 제한 주의)
    # Option 2: EC2 Spot Instance 시작하여 학습 위임
    # Option 3: SageMaker Training Job 시작

    logger.info(f"Training triggered with {pending_count} samples")

    # 메타데이터 업데이트 (학습 완료로 표시)
    update_metadata(0, training_triggered=True)

    return {
        'statusCode': 200,
        'body': json.dumps({
            'success': True,
            'message': 'Training triggered successfully',
            'trigger_type': trigger_type,
            'samples_count': pending_count,
            'timestamp': timestamp
        })
    }
