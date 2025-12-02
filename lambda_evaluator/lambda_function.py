"""
HairMe A/B Test Evaluator Lambda

매일 자동 실행되어 A/B 테스트 결과를 평가하고,
Challenger 모델이 유의미하게 우수한 경우 자동으로 승격합니다.

스케줄: 매일 UTC 15:00 (KST 00:00)

평가 기준:
- 최소 샘플 수: 100개 (Champion + Challenger 합계)
- 최소 개선율: 2% (positive_feedback_rate 기준)
- 승격 조건: challenger_wins + confidence >= medium

Author: HairMe ML Team
Date: 2025-12-02
"""

import json
import os
import io
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import traceback

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
S3_BUCKET = os.getenv('MLOPS_S3_BUCKET', 'hairme-mlops')
ANALYZE_LAMBDA_NAME = os.getenv('ANALYZE_LAMBDA_NAME', 'hairme-analyze')
# AWS_REGION은 Lambda 내장 환경변수 사용 (AWS_DEFAULT_REGION)
AWS_REGION = os.getenv('AWS_DEFAULT_REGION', os.getenv('AWS_REGION', 'ap-northeast-2'))
DYNAMODB_TABLE = os.getenv('DYNAMODB_TABLE_NAME', 'hairme-analysis')

# 평가 파라미터
MIN_SAMPLES = int(os.getenv('EVAL_MIN_SAMPLES', '100'))
MIN_IMPROVEMENT = float(os.getenv('EVAL_MIN_IMPROVEMENT', '0.02'))  # 2%
AUTO_PROMOTE = os.getenv('AUTO_PROMOTE', 'true').lower() == 'true'


@dataclass
class ABTestMetrics:
    """A/B 테스트 변형별 지표"""
    variant: str
    sample_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    positive_feedback_rate: float = 0.0
    avg_score_for_good: float = 0.0
    avg_score_for_bad: float = 0.0
    score_discrimination: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant': self.variant,
            'sample_count': self.sample_count,
            'positive_count': self.positive_count,
            'negative_count': self.negative_count,
            'positive_feedback_rate': round(self.positive_feedback_rate, 4),
            'avg_score_for_good': round(self.avg_score_for_good, 2),
            'avg_score_for_bad': round(self.avg_score_for_bad, 2),
            'score_discrimination': round(self.score_discrimination, 2)
        }


def get_s3_client():
    """S3 클라이언트"""
    import boto3
    return boto3.client('s3', region_name=AWS_REGION)


def get_lambda_client():
    """Lambda 클라이언트"""
    import boto3
    return boto3.client('lambda', region_name=AWS_REGION)


def get_dynamodb_table():
    """DynamoDB 테이블"""
    import boto3
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    return dynamodb.Table(DYNAMODB_TABLE)


def get_current_experiment_id() -> Optional[str]:
    """현재 실행 중인 실험 ID 조회"""
    lambda_client = get_lambda_client()

    try:
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        env_vars = response.get('Environment', {}).get('Variables', {})
        return env_vars.get('ABTEST_EXPERIMENT_ID', '')
    except Exception as e:
        logger.error(f"Failed to get experiment ID: {e}")
        return None


def get_current_ab_config() -> Dict[str, Any]:
    """현재 A/B 테스트 설정 조회"""
    lambda_client = get_lambda_client()

    try:
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        env_vars = response.get('Environment', {}).get('Variables', {})

        return {
            'enabled': env_vars.get('ABTEST_ENABLED', 'false').lower() == 'true',
            'experiment_id': env_vars.get('ABTEST_EXPERIMENT_ID', ''),
            'champion_version': env_vars.get('ABTEST_CHAMPION_VERSION', 'v6'),
            'challenger_version': env_vars.get('ABTEST_CHALLENGER_VERSION', ''),
            'challenger_percent': int(env_vars.get('ABTEST_CHALLENGER_PERCENT', '10'))
        }
    except Exception as e:
        logger.error(f"Failed to get AB config: {e}")
        return {}


def get_metrics_by_variant(experiment_id: str) -> Dict[str, ABTestMetrics]:
    """
    DynamoDB에서 실험 데이터를 조회하여 변형별 지표 계산
    """
    table = get_dynamodb_table()

    try:
        # DynamoDB Scan (실제 운영에서는 GSI 사용 권장)
        response = table.scan(
            FilterExpression='experiment_id = :exp_id AND attribute_exists(feedback_at)',
            ExpressionAttributeValues={
                ':exp_id': experiment_id
            },
            Limit=10000
        )

        items = response.get('Items', [])

        # 페이지네이션
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                FilterExpression='experiment_id = :exp_id AND attribute_exists(feedback_at)',
                ExpressionAttributeValues={
                    ':exp_id': experiment_id
                },
                ExclusiveStartKey=response['LastEvaluatedKey'],
                Limit=10000 - len(items)
            )
            items.extend(response.get('Items', []))

        logger.info(f"Retrieved {len(items)} items for experiment {experiment_id}")

        # 변형별 데이터 분류
        champion_data = []
        challenger_data = []

        for item in items:
            variant = item.get('ab_variant', 'champion')
            if variant == 'challenger':
                challenger_data.append(item)
            else:
                champion_data.append(item)

        champion_metrics = _calculate_metrics('champion', champion_data)
        challenger_metrics = _calculate_metrics('challenger', challenger_data)

        return {
            'champion': champion_metrics,
            'challenger': challenger_metrics
        }

    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        return {}


def _calculate_metrics(variant: str, items: List[Dict]) -> ABTestMetrics:
    """단일 변형에 대한 지표 계산"""
    metrics = ABTestMetrics(variant=variant)

    if not items:
        return metrics

    good_scores = []
    bad_scores = []

    for item in items:
        # 3개 스타일 각각에 대해 피드백 확인
        for i in range(1, 4):
            feedback_key = f'style_{i}_feedback'
            score_key = f'style_{i}_score'

            feedback = item.get(feedback_key)
            score = item.get(score_key)

            if feedback in ['good', 'like']:
                metrics.positive_count += 1
                if score is not None:
                    good_scores.append(float(score))
            elif feedback in ['bad', 'dislike']:
                metrics.negative_count += 1
                if score is not None:
                    bad_scores.append(float(score))

    metrics.sample_count = metrics.positive_count + metrics.negative_count

    if metrics.sample_count > 0:
        metrics.positive_feedback_rate = metrics.positive_count / metrics.sample_count

    if good_scores:
        metrics.avg_score_for_good = sum(good_scores) / len(good_scores)

    if bad_scores:
        metrics.avg_score_for_bad = sum(bad_scores) / len(bad_scores)

    if good_scores and bad_scores:
        metrics.score_discrimination = metrics.avg_score_for_good - metrics.avg_score_for_bad

    return metrics


def evaluate_experiment(metrics: Dict[str, ABTestMetrics]) -> Dict[str, Any]:
    """
    실험 결과 평가

    Returns:
        {
            'conclusion': 'challenger_wins' | 'champion_wins' | 'no_difference' | 'insufficient_data',
            'should_promote': bool,
            'improvement': float,
            'confidence': 'high' | 'medium' | 'low',
            'recommendation': str
        }
    """
    champion = metrics.get('champion')
    challenger = metrics.get('challenger')

    result = {
        'champion_metrics': champion.to_dict() if champion else None,
        'challenger_metrics': challenger.to_dict() if challenger else None,
        'conclusion': 'insufficient_data',
        'should_promote': False,
        'improvement': 0.0,
        'confidence': 'low',
        'recommendation': '',
        'evaluated_at': datetime.now(timezone.utc).isoformat()
    }

    # 데이터 검증
    if not champion or not challenger:
        result['recommendation'] = "변형별 데이터가 부족합니다."
        return result

    # 최소 샘플 수 확인
    total_samples = champion.sample_count + challenger.sample_count
    if total_samples < MIN_SAMPLES:
        result['recommendation'] = (
            f"최소 샘플 수({MIN_SAMPLES})를 충족하지 않습니다. "
            f"현재: Champion={champion.sample_count}, Challenger={challenger.sample_count}"
        )
        return result

    # 긍정 피드백 비율 비교
    rate_diff = challenger.positive_feedback_rate - champion.positive_feedback_rate
    result['improvement'] = round(rate_diff, 4)

    # 신뢰도 계산
    if total_samples >= 1000:
        result['confidence'] = 'high'
    elif total_samples >= 500:
        result['confidence'] = 'medium'
    else:
        result['confidence'] = 'low'

    # 결론 도출
    if rate_diff > MIN_IMPROVEMENT:
        result['conclusion'] = 'challenger_wins'
        result['should_promote'] = True
        result['recommendation'] = (
            f"Challenger가 {rate_diff*100:.1f}% 우수합니다. 승격을 권장합니다."
        )
    elif rate_diff < -MIN_IMPROVEMENT:
        result['conclusion'] = 'champion_wins'
        result['should_promote'] = False
        result['recommendation'] = (
            f"Champion이 {-rate_diff*100:.1f}% 우수합니다. 기존 모델 유지 권장."
        )
    else:
        result['conclusion'] = 'no_difference'
        result['should_promote'] = False
        result['recommendation'] = (
            f"유의미한 차이 없음 (차이: {rate_diff*100:.1f}%). 더 많은 데이터 필요."
        )

    logger.info(
        f"Evaluation result: {result['conclusion']} "
        f"(improvement={result['improvement']:.2%}, confidence={result['confidence']})"
    )

    return result


def backup_lambda_config() -> Optional[Dict[str, Any]]:
    """Lambda 환경변수 백업"""
    lambda_client = get_lambda_client()
    s3 = get_s3_client()

    try:
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        env_vars = response.get('Environment', {}).get('Variables', {})

        backup_key = f'config_backups/pre_promote_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=backup_key,
            Body=json.dumps(env_vars, indent=2),
            ContentType='application/json'
        )
        logger.info(f"Config backup saved: {backup_key}")

        return env_vars

    except Exception as e:
        logger.error(f"Failed to backup config: {e}")
        return None


def promote_challenger() -> bool:
    """
    Challenger를 Champion으로 승격

    - ABTEST_CHAMPION_VERSION = 기존 CHALLENGER_VERSION
    - ABTEST_CHALLENGER_VERSION = "" (비움)
    - ABTEST_EXPERIMENT_ID = "" (비움)
    """
    lambda_client = get_lambda_client()

    try:
        # 현재 설정 조회
        response = lambda_client.get_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME
        )
        current_env = response.get('Environment', {}).get('Variables', {})

        # 백업
        backup_lambda_config()

        # 승격: Challenger → Champion
        new_champion = current_env.get('ABTEST_CHALLENGER_VERSION', '')
        if not new_champion:
            logger.error("No challenger version to promote")
            return False

        current_env['ABTEST_CHAMPION_VERSION'] = new_champion
        current_env['ABTEST_CHALLENGER_VERSION'] = ''
        current_env['ABTEST_EXPERIMENT_ID'] = ''  # 실험 종료

        # Lambda 업데이트
        lambda_client.update_function_configuration(
            FunctionName=ANALYZE_LAMBDA_NAME,
            Environment={'Variables': current_env}
        )

        logger.info(f"Challenger promoted to Champion: {new_champion}")
        return True

    except Exception as e:
        logger.error(f"Failed to promote challenger: {e}")
        traceback.print_exc()
        return False


def save_evaluation_result(
    experiment_id: str,
    evaluation: Dict[str, Any],
    promoted: bool
):
    """평가 결과를 S3에 저장"""
    s3 = get_s3_client()

    try:
        result = {
            'experiment_id': experiment_id,
            'evaluation': evaluation,
            'promoted': promoted,
            'auto_promote_enabled': AUTO_PROMOTE,
            'evaluated_at': datetime.now(timezone.utc).isoformat()
        }

        key = f'experiments/{experiment_id}/evaluation_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.json'
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(result, indent=2, ensure_ascii=False),
            ContentType='application/json'
        )

        logger.info(f"Evaluation result saved: {key}")

    except Exception as e:
        logger.error(f"Failed to save evaluation result: {e}")


def run_evaluation_pipeline() -> Dict[str, Any]:
    """
    전체 평가 파이프라인 실행

    Returns:
        결과 딕셔너리
    """
    result = {
        'success': False,
        'experiment_id': None,
        'evaluation': None,
        'promoted': False,
        'steps_completed': []
    }

    try:
        # 1. 현재 A/B 테스트 설정 확인
        logger.info("Step 1: Checking AB test config")
        ab_config = get_current_ab_config()

        if not ab_config.get('enabled'):
            result['message'] = 'A/B test is not enabled'
            result['success'] = True  # 비활성화도 정상 상태
            return result

        experiment_id = ab_config.get('experiment_id')
        if not experiment_id:
            result['message'] = 'No active experiment'
            result['success'] = True
            return result

        challenger_version = ab_config.get('challenger_version')
        if not challenger_version:
            result['message'] = 'No challenger model configured'
            result['success'] = True
            return result

        result['experiment_id'] = experiment_id
        result['ab_config'] = ab_config
        result['steps_completed'].append('check_config')

        # 2. 지표 수집
        logger.info(f"Step 2: Collecting metrics for {experiment_id}")
        metrics = get_metrics_by_variant(experiment_id)

        if not metrics:
            result['message'] = 'Failed to collect metrics'
            return result

        result['steps_completed'].append('collect_metrics')

        # 3. 평가
        logger.info("Step 3: Evaluating experiment")
        evaluation = evaluate_experiment(metrics)
        result['evaluation'] = evaluation
        result['steps_completed'].append('evaluate')

        # 4. 자동 승격 (조건 충족 시)
        promoted = False
        if AUTO_PROMOTE and evaluation['should_promote']:
            logger.info("Step 4: Auto-promoting challenger")
            promoted = promote_challenger()
            result['promoted'] = promoted

            if promoted:
                result['steps_completed'].append('promote')
            else:
                result['steps_completed'].append('promote_failed')
        else:
            logger.info(
                f"Step 4: Skipping promotion "
                f"(auto_promote={AUTO_PROMOTE}, should_promote={evaluation['should_promote']})"
            )

        # 5. 결과 저장
        logger.info("Step 5: Saving evaluation result")
        save_evaluation_result(experiment_id, evaluation, promoted)
        result['steps_completed'].append('save_result')

        result['success'] = True
        result['message'] = (
            f"Evaluation completed: {evaluation['conclusion']}"
            + (f", promoted={promoted}" if evaluation['should_promote'] else "")
        )

        return result

    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {e}")
        traceback.print_exc()
        result['message'] = str(e)
        return result


def lambda_handler(event, context):
    """
    Lambda 핸들러

    Args:
        event: {
            "trigger_type": "scheduled" | "manual",
            "force_promote": false,  # true이면 조건 무시하고 승격
            "experiment_id": "exp_xxx"  # 특정 실험 평가 (선택)
        }

    Returns:
        평가 결과
    """
    logger.info("A/B Test Evaluator Lambda started")
    logger.info(f"Event: {json.dumps(event)}")

    trigger_type = event.get('trigger_type', 'unknown')
    force_promote = event.get('force_promote', False)
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        # 강제 승격 요청
        if force_promote:
            logger.info("Force promotion requested")
            backup_lambda_config()
            promoted = promote_challenger()

            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': promoted,
                    'message': 'Force promotion ' + ('succeeded' if promoted else 'failed'),
                    'trigger_type': trigger_type,
                    'timestamp': timestamp
                })
            }

        # 일반 평가 파이프라인
        result = run_evaluation_pipeline()

        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': result['success'],
                'message': result.get('message', ''),
                'trigger_type': trigger_type,
                'experiment_id': result.get('experiment_id'),
                'evaluation': result.get('evaluation'),
                'promoted': result.get('promoted', False),
                'steps_completed': result.get('steps_completed', []),
                'timestamp': timestamp
            })
        }

    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        traceback.print_exc()

        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'message': str(e),
                'trigger_type': trigger_type,
                'timestamp': timestamp
            })
        }
