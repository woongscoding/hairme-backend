#!/usr/bin/env bash
# hairme-analyze Lambda 워밍업 스케줄 생성 (EventBridge 규칙, 5분 간격)
#
# 목적: 콜드스타트(전체 요청의 ~14%, 첫 요청 8초)를 줄이기 위해 컨테이너를 주기적으로 깨우고
#       MediaPipe·torch 모델·S3 모델을 미리 로드한다. Lambda 핸들러가 {"warmup": true}
#       이벤트를 받으면 HTTP 스택을 거치지 않고 warm_up()만 실행한다 (main.py 참조).
# 비용: 월 ~8,640회 호출 × 수백 ms → 사실상 무료 (Provisioned Concurrency 대비).
#
# 필요 권한: events:PutRule, events:PutTargets, lambda:AddPermission
# 사용법:  bash scripts/setup_warmup_schedule.sh          # 생성/갱신
#          bash scripts/setup_warmup_schedule.sh --delete # 제거
set -euo pipefail
export MSYS_NO_PATHCONV=1

REGION="${AWS_REGION:-ap-northeast-2}"
FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-hairme-analyze}"
RULE_NAME="hairme-analyze-warmup"
RATE="${WARMUP_RATE:-rate(5 minutes)}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
RULE_ARN="arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}"

if [[ "${1:-}" == "--delete" ]]; then
  aws events remove-targets --rule "$RULE_NAME" --ids warmup --region "$REGION" || true
  aws events delete-rule --name "$RULE_NAME" --region "$REGION" || true
  aws lambda remove-permission --function-name "$FUNCTION_NAME" --statement-id "$RULE_NAME" --region "$REGION" || true
  echo "removed $RULE_NAME"
  exit 0
fi

aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$RATE" \
  --state ENABLED \
  --description "Keep hairme-analyze warm (MediaPipe/torch/S3 model preloaded)" \
  --region "$REGION" >/dev/null

# 이미 있으면 ResourceConflict가 나므로 무시
aws lambda add-permission \
  --function-name "$FUNCTION_NAME" \
  --statement-id "$RULE_NAME" \
  --action lambda:InvokeFunction \
  --principal events.amazonaws.com \
  --source-arn "$RULE_ARN" \
  --region "$REGION" >/dev/null 2>&1 || true

aws events put-targets \
  --rule "$RULE_NAME" \
  --targets "Id=warmup,Arn=${FUNCTION_ARN},Input='{\"warmup\": true}'" \
  --region "$REGION" >/dev/null

echo "ok: $RULE_NAME -> $FUNCTION_NAME every '$RATE' with payload {\"warmup\": true}"
echo "verify: aws lambda invoke --function-name $FUNCTION_NAME --cli-binary-format raw-in-base64-out --payload '{\"warmup\": true}' /dev/stdout --region $REGION"
