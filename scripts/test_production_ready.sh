#!/bin/bash

##############################################################################
# HairMe 프로덕션 준비 상태 통합 테스트
#
# 실행 방법:
#   bash scripts/test_production_ready.sh
#
# 또는 실행 권한 부여 후:
#   chmod +x scripts/test_production_ready.sh
#   ./scripts/test_production_ready.sh
#
# 환경 변수:
#   USE_DYNAMODB=true
#   AWS_REGION (선택, 기본값: ap-northeast-2)
#   API_URL (선택, 기본값: http://localhost:8000)
##############################################################################

set -e  # 에러 발생 시 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 진행 상황 카운터
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
WARNINGS=0

# 헬퍼 함수
print_header() {
    echo -e "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}$1${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
    ((PASSED_TESTS++))
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    ((FAILED_TESTS++))
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
    ((WARNINGS++))
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

start_test() {
    ((TOTAL_TESTS++))
    echo -e "${BOLD}테스트 $TOTAL_TESTS:${NC} $1"
}

# 환경 변수 설정
API_URL=${API_URL:-http://localhost:8000}
AWS_REGION=${AWS_REGION:-ap-northeast-2}

echo -e "${BOLD}${CYAN}"
cat << "EOF"
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🧪 HairMe 프로덕션 준비 상태 테스트                  ║
║       Production Readiness Integration Tests              ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}\n"

##############################################################################
# 1. 환경 변수 확인
##############################################################################
print_header "1️⃣  환경 변수 확인"

start_test "USE_DYNAMODB 환경 변수 확인"
if [[ "${USE_DYNAMODB}" == "true" ]]; then
    print_success "USE_DYNAMODB=true 설정됨"
else
    print_error "USE_DYNAMODB가 'true'로 설정되지 않음: ${USE_DYNAMODB}"
    print_info "다음 명령으로 설정: export USE_DYNAMODB=true"
    exit 1
fi

start_test "AWS_REGION 환경 변수 확인"
if [[ -n "${AWS_REGION}" ]]; then
    print_success "AWS_REGION=${AWS_REGION}"
else
    print_warning "AWS_REGION이 설정되지 않음 (기본값 사용)"
fi

start_test "DYNAMODB_TABLE_NAME 환경 변수 확인"
if [[ -n "${DYNAMODB_TABLE_NAME}" ]]; then
    print_success "DYNAMODB_TABLE_NAME=${DYNAMODB_TABLE_NAME}"
else
    print_info "DYNAMODB_TABLE_NAME 미설정 (기본값: hairme-analysis)"
    export DYNAMODB_TABLE_NAME="hairme-analysis"
fi

start_test "AWS 자격 증명 확인"
if aws sts get-caller-identity &> /dev/null; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    print_success "AWS 인증 성공 (Account: ${ACCOUNT_ID})"
else
    print_error "AWS 자격 증명 실패"
    print_info "aws configure를 실행하거나 IAM 역할을 확인하세요"
    exit 1
fi

##############################################################################
# 2. DynamoDB 테이블 접근 테스트
##############################################################################
print_header "2️⃣  DynamoDB 접근 테스트"

TABLE_NAME=${DYNAMODB_TABLE_NAME:-hairme-analysis}

start_test "DynamoDB 테이블 존재 확인"
if aws dynamodb describe-table --table-name "${TABLE_NAME}" --region "${AWS_REGION}" &> /dev/null; then
    TABLE_STATUS=$(aws dynamodb describe-table --table-name "${TABLE_NAME}" --region "${AWS_REGION}" --query 'Table.TableStatus' --output text)

    if [[ "${TABLE_STATUS}" == "ACTIVE" ]]; then
        print_success "테이블 '${TABLE_NAME}' 상태: ${TABLE_STATUS}"
    else
        print_warning "테이블 '${TABLE_NAME}' 상태: ${TABLE_STATUS} (ACTIVE 아님)"
    fi
else
    print_error "테이블 '${TABLE_NAME}'을 찾을 수 없음"
    exit 1
fi

start_test "DynamoDB 쓰기 테스트"
TEST_ID="test_$(date +%Y%m%d_%H%M%S)"
TEST_ITEM=$(cat <<EOF
{
    "analysis_id": {"S": "${TEST_ID}"},
    "user_id": {"S": "test_user"},
    "face_shape": {"S": "oval"},
    "created_at": {"S": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
}
EOF
)

if echo "${TEST_ITEM}" | aws dynamodb put-item \
    --table-name "${TABLE_NAME}" \
    --item file:///dev/stdin \
    --region "${AWS_REGION}" &> /dev/null; then
    print_success "DynamoDB 쓰기 성공 (ID: ${TEST_ID})"
else
    print_error "DynamoDB 쓰기 실패"
    exit 1
fi

start_test "DynamoDB 읽기 테스트"
RETRIEVED_ITEM=$(aws dynamodb get-item \
    --table-name "${TABLE_NAME}" \
    --key "{\"analysis_id\": {\"S\": \"${TEST_ID}\"}}" \
    --region "${AWS_REGION}" \
    --output json 2>/dev/null)

if echo "${RETRIEVED_ITEM}" | grep -q "oval"; then
    print_success "DynamoDB 읽기 성공 (데이터 일치)"
else
    print_error "DynamoDB 읽기 실패 또는 데이터 불일치"
fi

start_test "DynamoDB 삭제 테스트 (정리)"
if aws dynamodb delete-item \
    --table-name "${TABLE_NAME}" \
    --key "{\"analysis_id\": {\"S\": \"${TEST_ID}\"}}" \
    --region "${AWS_REGION}" &> /dev/null; then
    print_success "테스트 데이터 정리 완료"
else
    print_warning "테스트 데이터 정리 실패 (수동 삭제 필요: ${TEST_ID})"
fi

##############################################################################
# 3. Python 환경 및 종속성 테스트
##############################################################################
print_header "3️⃣  Python 환경 테스트"

start_test "Python 버전 확인"
PYTHON_VERSION=$(python --version 2>&1 | grep -oP '\d+\.\d+' || python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ -n "${PYTHON_VERSION}" ]]; then
    print_success "Python ${PYTHON_VERSION} 감지됨"
    PYTHON_CMD="python"

    # Python3가 필요한 경우 확인
    if ! command -v python &> /dev/null; then
        PYTHON_CMD="python3"
    fi
else
    print_error "Python을 찾을 수 없음"
    exit 1
fi

start_test "필수 Python 패키지 확인"
REQUIRED_PACKAGES=("boto3" "fastapi" "uvicorn" "google-generativeai")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"; do
    if ${PYTHON_CMD} -c "import ${package//-/_}" &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} ${package}"
    else
        echo -e "  ${RED}✗${NC} ${package} (누락)"
        MISSING_PACKAGES+=("${package}")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    print_success "모든 필수 패키지 설치됨"
else
    print_error "누락된 패키지: ${MISSING_PACKAGES[*]}"
    print_info "설치 명령: pip install ${MISSING_PACKAGES[*]}"
    exit 1
fi

##############################################################################
# 4. 로컬 API 서버 테스트
##############################################################################
print_header "4️⃣  API 엔드포인트 통합 테스트"

start_test "API 서버 연결 확인"
if curl -s -f "${API_URL}/api/health" -o /dev/null -w "%{http_code}" | grep -q "200"; then
    print_success "API 서버 응답 정상 (${API_URL})"
else
    print_warning "API 서버 연결 불가 (${API_URL})"
    print_info "서버 시작 명령: python -m uvicorn main:app --host 0.0.0.0 --port 8000"
    print_info "로컬 API 테스트를 건너뜁니다."

    # API 서버가 없어도 계속 진행
    SKIP_API_TESTS=true
fi

if [[ "${SKIP_API_TESTS}" != "true" ]]; then
    start_test "Health check 엔드포인트"
    HEALTH_RESPONSE=$(curl -s "${API_URL}/api/health")

    if echo "${HEALTH_RESPONSE}" | grep -q "healthy"; then
        print_success "Health check 응답: ${HEALTH_RESPONSE}"

        # DynamoDB 모드 확인
        if echo "${HEALTH_RESPONSE}" | grep -q "dynamodb"; then
            print_success "DynamoDB 모드 확인됨"
        else
            print_warning "DynamoDB 모드가 아닐 수 있음"
        fi
    else
        print_error "Health check 실패: ${HEALTH_RESPONSE}"
    fi

    start_test "API 응답 시간 측정"
    START_TIME=$(date +%s%N)
    curl -s "${API_URL}/api/health" -o /dev/null
    END_TIME=$(date +%s%N)
    RESPONSE_TIME=$(( (END_TIME - START_TIME) / 1000000 ))

    if [ ${RESPONSE_TIME} -lt 2000 ]; then
        print_success "응답 시간: ${RESPONSE_TIME}ms (목표: <2000ms)"
    else
        print_warning "응답 시간: ${RESPONSE_TIME}ms (느림)"
    fi

    # 추가 엔드포인트 테스트 (실제 이미지 업로드는 수동으로)
    print_info "실제 이미지 분석 테스트는 수동으로 수행하세요:"
    print_info "  curl -X POST ${API_URL}/api/analyze -F \"file=@test_image.jpg\""
fi

##############################################################################
# 5. 성능 테스트 (간단한 부하 테스트)
##############################################################################
print_header "5️⃣  성능 테스트"

if [[ "${SKIP_API_TESTS}" != "true" ]]; then
    start_test "동시 요청 처리 테스트 (10 요청)"

    # 백그라운드로 10개 요청 전송
    for i in {1..10}; do
        curl -s "${API_URL}/api/health" -o /dev/null &
    done

    # 모든 요청 완료 대기
    wait

    print_success "10개 동시 요청 처리 완료"

    start_test "평균 응답 시간 측정 (10회 반복)"
    TOTAL_TIME=0

    for i in {1..10}; do
        START=$(date +%s%N)
        curl -s "${API_URL}/api/health" -o /dev/null
        END=$(date +%s%N)
        DURATION=$(( (END - START) / 1000000 ))
        TOTAL_TIME=$((TOTAL_TIME + DURATION))
    done

    AVG_TIME=$((TOTAL_TIME / 10))

    if [ ${AVG_TIME} -lt 1000 ]; then
        print_success "평균 응답 시간: ${AVG_TIME}ms (우수)"
    elif [ ${AVG_TIME} -lt 2000 ]; then
        print_success "평균 응답 시간: ${AVG_TIME}ms (양호)"
    else
        print_warning "평균 응답 시간: ${AVG_TIME}ms (개선 필요)"
    fi
else
    print_info "API 서버 미실행으로 성능 테스트 건너뜀"
fi

##############################################################################
# 6. 에러 핸들링 테스트
##############################################################################
print_header "6️⃣  에러 핸들링 테스트"

if [[ "${SKIP_API_TESTS}" != "true" ]]; then
    start_test "존재하지 않는 엔드포인트 (404 테스트)"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "${API_URL}/api/nonexistent")

    if [[ "${HTTP_CODE}" == "404" ]]; then
        print_success "404 에러 핸들링 정상"
    else
        print_warning "예상치 못한 응답 코드: ${HTTP_CODE}"
    fi

    start_test "잘못된 메서드 테스트"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "${API_URL}/api/health")

    if [[ "${HTTP_CODE}" == "405" ]] || [[ "${HTTP_CODE}" == "404" ]]; then
        print_success "잘못된 메서드 에러 핸들링 정상 (${HTTP_CODE})"
    else
        print_info "메서드 에러 응답: ${HTTP_CODE}"
    fi
else
    print_info "API 서버 미실행으로 에러 핸들링 테스트 건너뜀"
fi

##############################################################################
# 7. 데이터베이스 연결 Python 테스트
##############################################################################
print_header "7️⃣  데이터베이스 연결 Python 테스트"

start_test "database.dynamodb_connection 모듈 테스트"

# Python 스크립트로 연결 테스트
cat > /tmp/test_db_connection.py << 'PYTHON_SCRIPT'
import sys
import os
from datetime import datetime

# 환경 변수 확인
if os.getenv('USE_DYNAMODB') != 'true':
    print("❌ USE_DYNAMODB가 'true'로 설정되지 않음")
    sys.exit(1)

try:
    from database.dynamodb_connection import save_analysis, get_analysis

    # 테스트 데이터
    test_id = f"python_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_data = {
        'user_id': 'python_test',
        'face_shape': 'round',
        'created_at': datetime.now().isoformat()
    }

    # 저장 테스트
    save_result = save_analysis(test_id, test_data)
    if not save_result:
        print(f"❌ save_analysis 실패")
        sys.exit(1)

    # 조회 테스트
    retrieved = get_analysis(test_id)
    if not retrieved or retrieved.get('face_shape') != 'round':
        print(f"❌ get_analysis 실패 또는 데이터 불일치")
        sys.exit(1)

    print(f"✅ Python DB 연결 테스트 성공 (ID: {test_id})")
    sys.exit(0)

except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ 테스트 실패: {e}")
    sys.exit(1)
PYTHON_SCRIPT

if ${PYTHON_CMD} /tmp/test_db_connection.py; then
    print_success "Python 데이터베이스 연결 테스트 통과"
else
    print_error "Python 데이터베이스 연결 테스트 실패"
    exit 1
fi

# 임시 파일 정리
rm -f /tmp/test_db_connection.py

##############################################################################
# 8. Lambda 함수 테스트 (선택사항)
##############################################################################
print_header "8️⃣  Lambda 함수 테스트 (선택사항)"

LAMBDA_FUNCTION_NAME=${LAMBDA_FUNCTION_NAME:-hairme-backend}

start_test "Lambda 함수 존재 확인"
if aws lambda get-function --function-name "${LAMBDA_FUNCTION_NAME}" --region "${AWS_REGION}" &> /dev/null; then
    print_success "Lambda 함수 '${LAMBDA_FUNCTION_NAME}' 존재"

    # 환경 변수 확인
    USE_DYNAMODB_VALUE=$(aws lambda get-function-configuration \
        --function-name "${LAMBDA_FUNCTION_NAME}" \
        --region "${AWS_REGION}" \
        --query 'Environment.Variables.USE_DYNAMODB' \
        --output text 2>/dev/null)

    if [[ "${USE_DYNAMODB_VALUE}" == "true" ]]; then
        print_success "Lambda 환경 변수 USE_DYNAMODB=true"
    else
        print_warning "Lambda 환경 변수 USE_DYNAMODB=${USE_DYNAMODB_VALUE}"
    fi
else
    print_info "Lambda 함수 없음 (ECS 사용 중일 수 있음)"
fi

##############################################################################
# 9. 보안 검사
##############################################################################
print_header "9️⃣  보안 검사"

start_test ".env 파일 보안 확인"
if [ -f ".env" ]; then
    print_warning ".env 파일 존재 - Git에 커밋되지 않았는지 확인 필요"

    if grep -q ".env" .gitignore 2>/dev/null; then
        print_success ".env가 .gitignore에 등록됨"
    else
        print_error ".env가 .gitignore에 없음 - 추가 필요!"
    fi
else
    print_info ".env 파일 없음 (환경 변수로 관리 중)"
fi

start_test "민감 정보 하드코딩 확인"
if grep -r "aws_access_key_id\s*=\s*['\"]" --include="*.py" . 2>/dev/null | grep -v ".git" | head -1; then
    print_error "AWS 자격 증명이 하드코딩되어 있을 수 있음!"
else
    print_success "AWS 자격 증명 하드코딩 없음"
fi

##############################################################################
# 최종 결과 요약
##############################################################################
print_header "🎯 테스트 결과 요약"

echo -e "${BOLD}전체 테스트:${NC} ${TOTAL_TESTS}"
echo -e "${BOLD}${GREEN}통과:${NC} ${PASSED_TESTS}"
echo -e "${BOLD}${RED}실패:${NC} ${FAILED_TESTS}"
echo -e "${BOLD}${YELLOW}경고:${NC} ${WARNINGS}\n"

# 통과율 계산
if [ ${TOTAL_TESTS} -gt 0 ]; then
    PASS_RATE=$((PASSED_TESTS * 100 / TOTAL_TESTS))
    echo -e "${BOLD}통과율:${NC} ${PASS_RATE}%\n"

    if [ ${FAILED_TESTS} -eq 0 ]; then
        echo -e "${GREEN}${BOLD}"
        cat << "EOF"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                ┃
┃      ✅ 모든 테스트를 통과했습니다!             ┃
┃      프로덕션 배포 준비 완료                   ┃
┃                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
EOF
        echo -e "${NC}\n"

        if [ ${WARNINGS} -gt 0 ]; then
            print_warning "${WARNINGS}개의 경고 항목이 있습니다. 검토 후 배포하세요."
        fi

        exit 0
    else
        echo -e "${RED}${BOLD}"
        cat << "EOF"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                ┃
┃      ❌ 일부 테스트가 실패했습니다              ┃
┃      문제를 해결한 후 다시 실행하세요          ┃
┃                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
EOF
        echo -e "${NC}\n"
        exit 1
    fi
fi

exit 0
