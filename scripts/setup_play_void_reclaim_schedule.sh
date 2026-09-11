#!/usr/bin/env bash
# Google Play 환불/취소 크레딧 회수 스케줄 생성 (EventBridge 규칙, 1일 간격)
#
# 목적: 환불/취소/차지백된 인앱결제의 크레딧을 회수한다 (보안 감사 H2).
#       Lambda 핸들러가 {"job": "reclaim_voided_purchases"} 이벤트를 받으면
#       HTTP 스택을 거치지 않고 PlayVoidReclaimService.run() 만 실행한다 (main.py 참조).
# 비용: 월 ~30회 호출 → 사실상 무료.
#
# 사전 조건: PLAY_PACKAGE_NAME 설정 + Secrets Manager `hairme-play-service-account`
#            (둘 중 하나라도 없으면 잡은 disabled 요약만 반환하고 아무 것도 하지 않는다)
#
# 필요 권한: events:PutRule, events:PutTargets, lambda:AddPermission
# 사용법:  bash scripts/setup_play_void_reclaim_schedule.sh          # 생성/갱신
#          bash scripts/setup_play_void_reclaim_schedule.sh --delete # 제거
set -euo pipefail
export MSYS_NO_PATHCONV=1

REGION="${AWS_REGION:-ap-northeast-2}"
FUNCTION_NAME="${LAMBDA_FUNCTION_NAME:-hairme-analyze}"
RULE_NAME="hairme-play-void-reclaim"
RATE="${VOID_RECLAIM_RATE:-rate(1 day)}"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
FUNCTION_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${FUNCTION_NAME}"
RULE_ARN="arn:aws:events:${REGION}:${ACCOUNT_ID}:rule/${RULE_NAME}"

if [[ "${1:-}" == "--delete" ]]; then
  aws events remove-targets --rule "$RULE_NAME" --ids reclaim --region "$REGION" || true
  aws events delete-rule --name "$RULE_NAME" --region "$REGION" || true
  aws lambda remove-permission --function-name "$FUNCTION_NAME" --statement-id "$RULE_NAME" --region "$REGION" || true
  echo "removed $RULE_NAME"
  exit 0
fi

aws events put-rule \
  --name "$RULE_NAME" \
  --schedule-expression "$RATE" \
  --state ENABLED \
  --description "Reclaim credits for refunded/voided Google Play purchases" \
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
  --targets "Id=reclaim,Arn=${FUNCTION_ARN},Input='{\"job\": \"reclaim_voided_purchases\"}'" \
  --region "$REGION" >/dev/null

echo "ok: $RULE_NAME -> $FUNCTION_NAME every '$RATE' with payload {\"job\": \"reclaim_voided_purchases\"}"
echo "verify: aws lambda invoke --function-name $FUNCTION_NAME --cli-binary-format raw-in-base64-out --payload '{\"job\": \"reclaim_voided_purchases\"}' /dev/stdout --region $REGION"
