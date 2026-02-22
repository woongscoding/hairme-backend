"""
DynamoDB connection and operations for HairMe Backend

This module provides DynamoDB operations that are compatible with the existing
MySQL-based interface, allowing seamless migration from RDS to DynamoDB.

Environment Variables:
    - AWS_REGION: AWS region for DynamoDB (default: ap-northeast-2)
    - DYNAMODB_TABLE_NAME: DynamoDB table name (default: hairme-analysis)
    - USE_DYNAMODB: Set to 'true' to enable DynamoDB (default: false)

Usage:
    from database.dynamodb_connection import (
        init_dynamodb,
        save_analysis,
        save_feedback,
        get_analysis,
        get_recent_analyses,
        get_feedback_stats
    )

    # Initialize connection
    if init_dynamodb():
        # Save analysis result
        analysis_id = save_analysis({
            'face_shape': '계란형',
            'personal_color': '봄웜',
            'recommendations': [...],
            ...
        })

        # Save feedback
        save_feedback(analysis_id, 1, 'good', True)

        # Retrieve analysis
        result = get_analysis(analysis_id)
"""

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, List, Any
from decimal import Decimal

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

try:
    from core.cache import redis_client
except ImportError:
    redis_client = None

from config.settings import settings
from core.logging import logger, log_structured

# Global DynamoDB resources
dynamodb_resource = None
dynamodb_table = None
dynamodb_enabled = False


def _convert_floats_to_decimal(obj: Any) -> Any:
    """
    Convert Python floats to Decimal for DynamoDB compatibility

    DynamoDB doesn't support Python float type directly, requires Decimal.

    Args:
        obj: Object to convert (can be dict, list, float, or other types)

    Returns:
        Converted object with floats replaced by Decimals
    """
    if isinstance(obj, float):
        return Decimal(str(obj))
    elif isinstance(obj, dict):
        return {k: _convert_floats_to_decimal(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_floats_to_decimal(item) for item in obj]
    return obj


def _convert_decimals_to_float(obj: Any) -> Any:
    """
    Convert Decimal to Python float for JSON serialization

    Args:
        obj: Object to convert (can be dict, list, Decimal, or other types)

    Returns:
        Converted object with Decimals replaced by floats
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _convert_decimals_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_decimals_to_float(item) for item in obj]
    return obj


def init_dynamodb() -> bool:
    """
    Initialize DynamoDB connection

    Reads AWS_REGION and DYNAMODB_TABLE_NAME from environment variables.

    Returns:
        bool: True if connection successful, False otherwise
    """
    global dynamodb_resource, dynamodb_table, dynamodb_enabled

    if not BOTO3_AVAILABLE:
        logger.error("❌ boto3 not installed. Install: pip install boto3")
        return False

    # Check if DynamoDB is enabled
    use_dynamodb = os.getenv("USE_DYNAMODB", "false").lower() == "true"
    if not use_dynamodb:
        logger.info("ℹ️  DynamoDB disabled (USE_DYNAMODB=false)")
        return False

    aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
    table_name = os.getenv("DYNAMODB_TABLE_NAME", "hairme-analysis")

    try:
        logger.info(f"🔌 Connecting to DynamoDB in {aws_region}...")

        # Create config with timeouts
        config = Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 3})

        # Create DynamoDB resource
        dynamodb_resource = boto3.resource(
            "dynamodb", region_name=aws_region, config=config
        )
        dynamodb_table = dynamodb_resource.Table(table_name)

        # Verify table exists by describing it
        dynamodb_table.load()

        dynamodb_enabled = True

        logger.info(f"✅ DynamoDB 연결 성공: {table_name}")
        log_structured(
            "dynamodb_connected",
            {"table_name": table_name, "region": aws_region, "status": "active"},
        )

        return True

    except NoCredentialsError:
        logger.error("❌ AWS credentials not configured. Run: aws configure")
        dynamodb_enabled = False
        return False

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ResourceNotFoundException":
            logger.error(
                f"❌ Table '{table_name}' not found. Run: ./scripts/create_dynamodb_table.sh"
            )
        else:
            logger.error(
                f"❌ DynamoDB connection failed: {e.response['Error']['Message']}"
            )
        dynamodb_enabled = False
        return False

    except Exception as e:
        logger.error(f"❌ DynamoDB initialization failed: {str(e)}")
        dynamodb_enabled = False
        return False


def save_analysis(data: Dict[str, Any]) -> Optional[str]:
    """
    Save analysis result to DynamoDB

    This function provides MySQL-compatible interface for saving analysis data.
    Maps AnalysisHistory model fields to DynamoDB attributes.

    Args:
        data: Dictionary containing analysis data with keys:
            - face_shape (str): Detected face shape
            - personal_color (str): Personal color analysis result
            - recommendations (list): Recommended hairstyles
            - processing_time (float): Analysis processing time
            - detection_method (str): Detection method used
            - image_hash (str): SHA-256 hash of image
            - user_id (str, optional): User ID (default: 'anonymous')
            - opencv_* (float, optional): OpenCV measurements
            - mediapipe_* (float/bool, optional): MediaPipe measurements
            - recommended_styles (list, optional): Detailed style recommendations

    Returns:
        str: Generated analysis_id (UUID) if successful, None otherwise

    Example:
        >>> analysis_id = save_analysis({
        ...     'face_shape': '계란형',
        ...     'personal_color': '봄웜',
        ...     'recommendations': [{'style_name': '레이어드 컷', 'reason': '...'}],
        ...     'processing_time': 2.5,
        ...     'detection_method': 'mediapipe',
        ...     'image_hash': 'abc123...'
        ... })
        >>> print(analysis_id)
        '550e8400-e29b-41d4-a716-446655440000'
    """
    if not dynamodb_enabled or dynamodb_table is None:
        logger.error("❌ DynamoDB not initialized")
        return None

    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Build DynamoDB item
        item = {
            # Primary Key & GSI
            "analysis_id": analysis_id,
            "entity_type": "ANALYSIS",  # For GSI partition key
            "created_at": now,
            # Basic Information
            "user_id": data.get("user_id", "anonymous"),
            "image_hash": data.get("image_hash", ""),
            "face_shape": data.get("face_shape"),
            "personal_color": data.get("personal_color"),
            "recommendations": data.get("recommendations", []),
            "processing_time": data.get("processing_time", 0.0),
            "detection_method": data.get("detection_method", "unknown"),
        }

        # OpenCV Measurements (10 fields)
        opencv_fields = [
            "opencv_face_ratio",
            "opencv_forehead_ratio",
            "opencv_cheekbone_ratio",
            "opencv_jaw_ratio",
            "opencv_prediction",
            "opencv_confidence",
            "opencv_gemini_agreement",
            "opencv_upper_face_ratio",
            "opencv_middle_face_ratio",
            "opencv_lower_face_ratio",
        ]
        for field in opencv_fields:
            if field in data and data[field] is not None:
                item[field] = data[field]

        # MediaPipe Measurements (10 fields)
        mediapipe_fields = [
            "mediapipe_face_ratio",
            "mediapipe_forehead_width",
            "mediapipe_cheekbone_width",
            "mediapipe_jaw_width",
            "mediapipe_forehead_ratio",
            "mediapipe_jaw_ratio",
            "mediapipe_ITA_value",
            "mediapipe_hue_value",
            "mediapipe_confidence",
            "mediapipe_features_complete",
        ]
        for field in mediapipe_fields:
            if field in data and data[field] is not None:
                item[field] = data[field]

        # Feedback Data (initially null/false)
        item.update(
            {
                "recommended_styles": data.get("recommended_styles", []),
                "style_1_feedback": None,
                "style_2_feedback": None,
                "style_3_feedback": None,
                "style_4_feedback": None,
                "style_5_feedback": None,
                "style_1_naver_clicked": False,
                "style_2_naver_clicked": False,
                "style_3_naver_clicked": False,
                "style_4_naver_clicked": False,
                "style_5_naver_clicked": False,
                "feedback_at": None,
            }
        )

        # A/B 테스트 정보 (model_version, experiment_id, ab_variant)
        if data.get("model_version"):
            item["model_version"] = data.get("model_version")
        if data.get("experiment_id"):
            item["experiment_id"] = data.get("experiment_id")
        if data.get("ab_variant"):
            item["ab_variant"] = data.get("ab_variant")

        # 스타일별 점수 저장 (A/B 테스트 평가용)
        for i in range(1, 6):
            score_key = f"style_{i}_score"
            if data.get(score_key) is not None:
                item[score_key] = data.get(score_key)

        # Convert floats to Decimal for DynamoDB
        item = _convert_floats_to_decimal(item)

        # Save to DynamoDB
        dynamodb_table.put_item(Item=item)

        logger.info(f"✅ DynamoDB 저장 성공 (ID: {analysis_id})")
        log_structured(
            "dynamodb_saved",
            {
                "analysis_id": analysis_id,
                "face_shape": data.get("face_shape"),
                "detection_method": data.get("detection_method"),
                "mediapipe_enabled": data.get("mediapipe_features_complete", False),
            },
        )

        return analysis_id

    except ClientError as e:
        logger.error(f"❌ DynamoDB 저장 실패: {e.response['Error']['Message']}")
        return None

    except Exception as e:
        logger.error(f"❌ DynamoDB 저장 실패: {str(e)}")
        return None


def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve analysis result by ID

    Args:
        analysis_id: UUID of the analysis record

    Returns:
        dict: Analysis data with all attributes, or None if not found

    Example:
        >>> result = get_analysis('550e8400-e29b-41d4-a716-446655440000')
        >>> print(result['face_shape'])
        '계란형'
    """
    if not dynamodb_enabled or dynamodb_table is None:
        logger.error("❌ DynamoDB not initialized")
        return None

    try:
        response = dynamodb_table.get_item(Key={"analysis_id": analysis_id})

        if "Item" not in response:
            logger.warning(f"⚠️  Analysis not found: {analysis_id}")
            return None

        item = response["Item"]

        # Convert Decimals to floats for compatibility
        item = _convert_decimals_to_float(item)

        logger.info(f"✅ Analysis 조회 성공: {analysis_id}")
        return item

    except ClientError as e:
        logger.error(f"❌ DynamoDB 조회 실패: {e.response['Error']['Message']}")
        return None

    except Exception as e:
        logger.error(f"❌ DynamoDB 조회 실패: {str(e)}")
        return None


def save_feedback(
    analysis_id: str, style_index: int, feedback: str, naver_clicked: bool
) -> bool:
    """
    Save user feedback for a specific style recommendation

    Args:
        analysis_id: UUID of the analysis record
        style_index: Style index (1, 2, or 3)
        feedback: Feedback value ('good' or 'bad')
        naver_clicked: Whether user clicked Naver search link

    Returns:
        bool: True if successful, False otherwise

    Example:
        >>> save_feedback('550e8400-...', 1, 'good', True)
        True
    """
    if not dynamodb_enabled or dynamodb_table is None:
        logger.error("❌ DynamoDB not initialized")
        return False

    if style_index not in [1, 2, 3, 4, 5]:
        logger.error(f"❌ Invalid style_index: {style_index} (must be 1-5)")
        return False

    try:
        feedback_field = f"style_{style_index}_feedback"
        clicked_field = f"style_{style_index}_naver_clicked"

        response = dynamodb_table.update_item(
            Key={"analysis_id": analysis_id},
            UpdateExpression=f"""
                SET {feedback_field} = :feedback,
                    {clicked_field} = :clicked,
                    feedback_at = :timestamp
            """,
            ExpressionAttributeValues={
                ":feedback": feedback,
                ":clicked": naver_clicked,
                ":timestamp": datetime.now(timezone.utc).isoformat(),
            },
            ReturnValues="UPDATED_NEW",
        )

        logger.info(
            f"✅ Feedback 저장 성공: analysis_id={analysis_id}, "
            f"style={style_index}, feedback={feedback}, clicked={naver_clicked}"
        )

        log_structured(
            "feedback_saved",
            {
                "analysis_id": analysis_id,
                "style_index": style_index,
                "feedback": feedback,
                "naver_clicked": naver_clicked,
            },
        )

        # ========== Invalidate Statistics Cache ==========
        if redis_client:
            try:
                redis_client.delete("feedback_stats")
                logger.info("✅ 통계 캐시 무효화 완료")
            except Exception as e:
                logger.warning(f"⚠️ 캐시 무효화 실패: {str(e)}")

        # ========== MLOps: S3 피드백 저장 (학습용) ==========
        try:
            _save_feedback_to_s3(analysis_id, style_index, feedback)
        except Exception as e:
            # S3 저장 실패해도 DynamoDB 저장은 성공으로 처리
            logger.warning(f"⚠️ S3 피드백 저장 실패 (MLOps): {str(e)}")

        return True

    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        if error_code == "ConditionalCheckFailedException":
            logger.error(f"❌ Analysis not found: {analysis_id}")
        else:
            logger.error(f"❌ Feedback 저장 실패: {e.response['Error']['Message']}")
        return False

    except Exception as e:
        logger.error(f"❌ Feedback 저장 실패: {str(e)}")
        return False


def get_recent_analyses(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Retrieve recent analysis results (ordered by created_at descending)

    Uses the created_at-index GSI for efficient time-based queries.

    Args:
        limit: Maximum number of results to return (default: 10)

    Returns:
        list: List of analysis dictionaries, newest first

    Example:
        >>> recent = get_recent_analyses(5)
        >>> for analysis in recent:
        ...     print(analysis['analysis_id'], analysis['created_at'])
    """
    if not dynamodb_enabled or dynamodb_table is None:
        logger.error("❌ DynamoDB not initialized")
        return []

    try:
        response = dynamodb_table.query(
            IndexName="created_at-index",
            KeyConditionExpression="entity_type = :type",
            ExpressionAttributeValues={":type": "ANALYSIS"},
            ScanIndexForward=False,  # Descending order (newest first)
            Limit=limit,
        )

        items = response.get("Items", [])

        # Convert Decimals to floats
        items = [_convert_decimals_to_float(item) for item in items]

        logger.info(f"✅ 최근 분석 {len(items)}개 조회 성공")
        return items

    except ClientError as e:
        logger.error(f"❌ DynamoDB 조회 실패: {e.response['Error']['Message']}")
        return []

    except Exception as e:
        logger.error(f"❌ DynamoDB 조회 실패: {str(e)}")
        return []


def get_feedback_stats() -> Dict[str, Any]:
    """
    Get feedback statistics from all analysis records

    Returns:
        dict: Statistics including:
            - total_analysis: Total number of analysis records
            - total_feedback: Number of records with feedback
            - like_counts: Likes per style (style_1, style_2, style_3)
            - dislike_counts: Dislikes per style
            - recent_feedbacks: Latest 5 feedback records

    Example:
        >>> stats = get_feedback_stats()
        >>> print(stats['total_analysis'])
        1523
        >>> print(stats['like_counts']['style_1'])
        342
    """
    if not dynamodb_enabled or dynamodb_table is None:
        logger.error("❌ DynamoDB not initialized")
        return {
            "success": False,
            "total_analysis": 0,
            "total_feedback": 0,
            "like_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "dislike_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "recent_feedbacks": [],
        }

    try:
        # ========== Redis Cache Layer (5-minute TTL) ==========
        # Reduces expensive scan() operations
        cache_key = "feedback_stats"

        if redis_client:
            try:
                cached_stats = redis_client.get(cache_key)
                if cached_stats:
                    logger.info("✅ 통계 캐시 히트 (DynamoDB scan 생략)")
                    return json.loads(cached_stats)
            except Exception as e:
                logger.warning(f"⚠️ Redis 캐시 조회 실패: {str(e)}")

        # ========== DynamoDB Scan (Cache Miss) ==========
        # TODO (Long-term): Replace with DynamoDB Streams + Lambda for real-time aggregation
        # or use separate aggregate table (feedback-stats) to avoid full table scan
        logger.info("🔍 통계 캐시 미스, DynamoDB scan 시작...")

        response = dynamodb_table.scan()
        items = response.get("Items", [])

        # Handle pagination if needed
        while "LastEvaluatedKey" in response:
            response = dynamodb_table.scan(
                ExclusiveStartKey=response["LastEvaluatedKey"]
            )
            items.extend(response.get("Items", []))

        total_analysis = len(items)
        feedback_records = [
            item for item in items if item.get("feedback_at") is not None
        ]
        total_feedback = len(feedback_records)

        # Count likes and dislikes
        like_counts = {
            "style_1": 0,
            "style_2": 0,
            "style_3": 0,
            "style_4": 0,
            "style_5": 0,
        }
        dislike_counts = {
            "style_1": 0,
            "style_2": 0,
            "style_3": 0,
            "style_4": 0,
            "style_5": 0,
        }

        for item in feedback_records:
            for i in [1, 2, 3, 4, 5]:
                feedback_field = f"style_{i}_feedback"
                feedback_value = item.get(feedback_field)

                if feedback_value == "like" or feedback_value == "good":
                    like_counts[f"style_{i}"] += 1
                elif feedback_value == "dislike" or feedback_value == "bad":
                    dislike_counts[f"style_{i}"] += 1

        # Get recent 5 feedbacks (sorted by feedback_at)
        sorted_feedbacks = sorted(
            feedback_records, key=lambda x: x.get("feedback_at", ""), reverse=True
        )[:5]

        recent_data = []
        for item in sorted_feedbacks:
            recent_data.append(
                {
                    "id": item.get("analysis_id"),
                    "face_shape": item.get("face_shape"),
                    "personal_color": item.get("personal_color"),
                    "style_1_feedback": item.get("style_1_feedback"),
                    "style_2_feedback": item.get("style_2_feedback"),
                    "style_3_feedback": item.get("style_3_feedback"),
                    "style_4_feedback": item.get("style_4_feedback"),
                    "style_5_feedback": item.get("style_5_feedback"),
                    "style_1_naver_clicked": item.get("style_1_naver_clicked", False),
                    "style_2_naver_clicked": item.get("style_2_naver_clicked", False),
                    "style_3_naver_clicked": item.get("style_3_naver_clicked", False),
                    "style_4_naver_clicked": item.get("style_4_naver_clicked", False),
                    "style_5_naver_clicked": item.get("style_5_naver_clicked", False),
                    "feedback_at": item.get("feedback_at"),
                    "created_at": item.get("created_at"),
                }
            )

        logger.info(f"📊 통계 조회: 전체 {total_analysis}개, 피드백 {total_feedback}개")

        result = {
            "success": True,
            "total_analysis": total_analysis,
            "total_feedback": total_feedback,
            "like_counts": like_counts,
            "dislike_counts": dislike_counts,
            "recent_feedbacks": recent_data,
        }

        # ========== Save to Redis Cache (TTL: 5 minutes) ==========
        if redis_client:
            try:
                redis_client.setex(
                    cache_key,
                    300,  # 5 minutes TTL
                    json.dumps(result, ensure_ascii=False),
                )
                logger.info("✅ 통계 캐시 저장 완료 (TTL: 5분)")
            except Exception as e:
                logger.warning(f"⚠️ Redis 캐시 저장 실패: {str(e)}")

        return result

    except ClientError as e:
        logger.error(f"❌ 통계 조회 실패: {e.response['Error']['Message']}")
        return {
            "success": False,
            "total_analysis": 0,
            "total_feedback": 0,
            "like_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "dislike_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "recent_feedbacks": [],
        }

    except Exception as e:
        logger.error(f"❌ 통계 조회 실패: {str(e)}")
        return {
            "success": False,
            "total_analysis": 0,
            "total_feedback": 0,
            "like_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "dislike_counts": {
                "style_1": 0,
                "style_2": 0,
                "style_3": 0,
                "style_4": 0,
                "style_5": 0,
            },
            "recent_feedbacks": [],
        }


# ========== MLOps: S3 피드백 저장 헬퍼 ==========


def _save_feedback_to_s3(analysis_id: str, style_index: int, feedback: str):
    """
    피드백을 S3에 저장 (ML 학습용)

    DynamoDB에서 분석 데이터를 가져와 S3에 NPZ 형식으로 저장합니다.
    MLOps 파이프라인에서 이 데이터를 사용하여 모델을 재학습합니다.

    Args:
        analysis_id: 분석 ID
        style_index: 스타일 인덱스 (1, 2, 3)
        feedback: 피드백 ('good' or 'bad')
    """
    import os

    # MLOps 활성화 여부 확인
    mlops_enabled = os.getenv("MLOPS_ENABLED", "false").lower() == "true"
    if not mlops_enabled:
        return

    try:
        # 1. 분석 데이터 조회
        analysis = get_analysis(analysis_id)
        if not analysis:
            logger.warning(f"⚠️ S3 저장 건너뜀: 분석 데이터 없음 ({analysis_id})")
            return

        # 2. 필요한 데이터 추출
        face_shape = analysis.get("face_shape", "")
        skin_tone = analysis.get("personal_color", "")

        # recommended_styles에서 hairstyle_id 추출
        recommended_styles = analysis.get("recommended_styles", [])
        if style_index <= len(recommended_styles):
            style_data = recommended_styles[style_index - 1]
            hairstyle_id = style_data.get("hairstyle_id")

            # hairstyle_id가 None인 경우: 트렌드 스타일이면 별도 처리
            if hairstyle_id is None:
                style_name = style_data.get("style_name", "")
                source = style_data.get("source", "")

                if source == "trending" and style_name:
                    # 트렌드 스타일: on-the-fly 임베딩 생성 후 S3 저장
                    try:
                        # MediaPipe 측정값 추출 (있으면)
                        _face_features = None
                        _skin_features = None
                        if analysis.get("mediapipe_face_ratio") is not None:
                            _face_features = [
                                analysis.get("mediapipe_face_ratio", 1.2),
                                analysis.get("mediapipe_forehead_width", 400),
                                analysis.get("mediapipe_cheekbone_width", 500),
                                analysis.get("mediapipe_jaw_width", 400),
                                analysis.get("mediapipe_forehead_ratio", 0.8),
                                analysis.get("mediapipe_jaw_ratio", 0.8),
                            ]
                        if analysis.get("mediapipe_ITA_value") is not None:
                            _skin_features = [
                                analysis.get("mediapipe_ITA_value", 70),
                                analysis.get("mediapipe_hue_value", 15),
                            ]

                        from services.mlops.s3_feedback_store import (
                            get_s3_feedback_store,
                        )

                        s3_store = get_s3_feedback_store()
                        result = s3_store.save_trending_feedback(
                            analysis_id=analysis_id,
                            face_shape=face_shape,
                            skin_tone=skin_tone,
                            style_name=style_name,
                            feedback=feedback,
                            face_features=_face_features,
                            skin_features=_skin_features,
                        )
                        if result.get("success"):
                            logger.info(f"트렌드 피드백 S3 저장 완료: {style_name}")
                    except Exception as e:
                        logger.warning(f"트렌드 피드백 S3 저장 실패: {e}")
                    return

                logger.warning(f"⚠️ hairstyle_id 없음 - style_name: {style_name}")
                return
        else:
            logger.warning(f"⚠️ S3 저장 건너뜀: style_index 범위 초과")
            return

        # 3. MediaPipe 측정값 추출 (있으면)
        face_features = None
        skin_features = None

        # MediaPipe 측정값이 저장되어 있으면 사용
        if analysis.get("mediapipe_face_ratio") is not None:
            face_features = [
                analysis.get("mediapipe_face_ratio", 1.2),
                analysis.get("mediapipe_forehead_width", 400),
                analysis.get("mediapipe_cheekbone_width", 500),
                analysis.get("mediapipe_jaw_width", 400),
                analysis.get("mediapipe_forehead_ratio", 0.8),
                analysis.get("mediapipe_jaw_ratio", 0.8),
            ]

        if analysis.get("mediapipe_ITA_value") is not None:
            skin_features = [
                analysis.get("mediapipe_ITA_value", 70),
                analysis.get("mediapipe_hue_value", 15),
            ]

        # 4. S3에 저장
        from services.mlops.s3_feedback_store import get_s3_feedback_store

        s3_store = get_s3_feedback_store()
        result = s3_store.save_feedback(
            analysis_id=analysis_id,
            face_shape=face_shape,
            skin_tone=skin_tone,
            hairstyle_id=hairstyle_id,
            feedback=feedback,
            face_features=face_features,
            skin_features=skin_features,
        )

        if result.get("success"):
            pending_count = result.get("pending_count", 0)
            logger.info(f"✅ S3 피드백 저장 완료 (pending: {pending_count})")

            # 재학습 트리거 확인
            if result.get("should_trigger_training"):
                from services.mlops.training_trigger import get_training_trigger

                trigger = get_training_trigger()
                trigger.check_and_trigger(pending_count)

    except ImportError as e:
        # MLOps 모듈이 없으면 무시 (선택적 기능)
        logger.debug(f"MLOps 모듈 없음: {e}")
    except Exception as e:
        logger.warning(f"⚠️ S3 피드백 저장 실패: {e}")
