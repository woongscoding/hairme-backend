#!/usr/bin/env bash
# AWS 월 비용 예산 + 알림 (수익화 v2.1 Phase 0)
#
# 실제 사용액이 예산의 80%, 100% 를 넘으면, 예측액이 100% 를 넘으면 메일을 보낸다.
# Gemini 비용은 Google 청구라 여기 잡히지 않는다 - Gemini 쪽은 DAILY_SYNTHESIS_BUDGET
# (비로그인 일 호출 상한)과 Google Cloud 예산 알림으로 따로 막는다.
#
# 사용법:
#   scripts/setup_aws_budget.sh <월 예산 USD> <알림 이메일>           # dry-run (명령만 출력)
#   scripts/setup_aws_budget.sh <월 예산 USD> <알림 이메일> --apply   # 실제 생성
#
# 같은 이름의 예산이 이미 있으면 create-budget 이 DuplicateRecordException 으로 실패한다.
# 금액만 바꿀 때는 콘솔이나 `aws budgets update-budget` 을 쓴다.
set -euo pipefail

LIMIT_USD="${1:?월 예산(USD)을 넣으세요 (예: 30)}"
EMAIL="${2:?알림 받을 이메일을 넣으세요}"
APPLY="${3:-}"
BUDGET_NAME="hairme-monthly"

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

BUDGET=$(cat <<EOF
{"BudgetName":"${BUDGET_NAME}","BudgetLimit":{"Amount":"${LIMIT_USD}","Unit":"USD"},
 "TimeUnit":"MONTHLY","BudgetType":"COST"}
EOF
)

notification() {
  # $1 = ACTUAL|FORECASTED, $2 = 임계 퍼센트
  cat <<EOF
{"Notification":{"NotificationType":"$1","ComparisonOperator":"GREATER_THAN",
  "Threshold":$2,"ThresholdType":"PERCENTAGE"},
 "Subscribers":[{"SubscriptionType":"EMAIL","Address":"${EMAIL}"}]}
EOF
}

NOTIFICATIONS="[$(notification ACTUAL 80),$(notification ACTUAL 100),$(notification FORECASTED 100)]"

echo "계정: ${ACCOUNT_ID}, 예산: ${BUDGET_NAME} = \$${LIMIT_USD}/월, 알림: ${EMAIL}"

if [ "${APPLY}" != "--apply" ]; then
  echo "(dry-run) 실행될 명령:"
  echo "aws budgets create-budget --account-id ${ACCOUNT_ID} --budget '${BUDGET}' --notifications-with-subscribers '${NOTIFICATIONS}'"
  exit 0
fi

aws budgets create-budget \
  --account-id "${ACCOUNT_ID}" \
  --budget "${BUDGET}" \
  --notifications-with-subscribers "${NOTIFICATIONS}"
echo "✅ 예산 생성 완료 (구독 확인 메일은 오지 않는다 - 바로 활성)"
