#!/bin/bash

# HairMe v20 API 테스트 스크립트

# 색상 코드
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# API URL 설정 (환경에 맞게 변경)
API_URL="${API_URL:-http://localhost:8000}"

echo "========================================="
echo "HairMe v20 API 테스트"
echo "API URL: $API_URL"
echo "========================================="
echo ""

# 1. 헬스체크
echo -e "${YELLOW}[1/4] 헬스체크...${NC}"
HEALTH_RESPONSE=$(curl -s $API_URL/api/health)
echo $HEALTH_RESPONSE | jq '.'

if echo $HEALTH_RESPONSE | jq -e '.feedback_system == "enabled"' > /dev/null; then
    echo -e "${GREEN}✅ 피드백 시스템 활성화 확인${NC}"
else
    echo -e "${RED}❌ 피드백 시스템 미활성화${NC}"
fi
echo ""

# 2. 루트 엔드포인트
echo -e "${YELLOW}[2/4] 루트 엔드포인트...${NC}"
ROOT_RESPONSE=$(curl -s $API_URL/)
echo $ROOT_RESPONSE | jq '.'

if echo $ROOT_RESPONSE | jq -e '.version == "20.0.0"' > /dev/null; then
    echo -e "${GREEN}✅ v20 버전 확인${NC}"
else
    echo -e "${RED}❌ 버전 불일치${NC}"
fi
echo ""

# 3. 얼굴 분석 (테스트 이미지 필요)
echo -e "${YELLOW}[3/4] 얼굴 분석...${NC}"
if [ -f "test_face.jpg" ]; then
    ANALYZE_RESPONSE=$(curl -s -X POST $API_URL/api/analyze \
        -F "file=@test_face.jpg")
    
    echo $ANALYZE_RESPONSE | jq '.'
    
    # analysis_id 추출
    ANALYSIS_ID=$(echo $ANALYZE_RESPONSE | jq -r '.analysis_id')
    
    if [ "$ANALYSIS_ID" != "null" ] && [ ! -z "$ANALYSIS_ID" ]; then
        echo -e "${GREEN}✅ analysis_id 확인: $ANALYSIS_ID${NC}"
        
        # 4. 피드백 제출
        echo ""
        echo -e "${YELLOW}[4/4] 피드백 제출...${NC}"
        
        FEEDBACK_RESPONSE=$(curl -s -X POST $API_URL/api/feedback \
            -H "Content-Type: application/json" \
            -d "{
                \"analysis_id\": $ANALYSIS_ID,
                \"style_index\": 1,
                \"feedback\": \"like\",
                \"naver_clicked\": true
            }")
        
        echo $FEEDBACK_RESPONSE | jq '.'
        
        if echo $FEEDBACK_RESPONSE | jq -e '.success == true' > /dev/null; then
            echo -e "${GREEN}✅ 피드백 제출 성공${NC}"
        else
            echo -e "${RED}❌ 피드백 제출 실패${NC}"
        fi
    else
        echo -e "${RED}❌ analysis_id 없음${NC}"
    fi
else
    echo -e "${RED}❌ test_face.jpg 파일이 없습니다${NC}"
    echo "테스트 이미지를 준비하고 다시 실행하세요"
fi

echo ""
echo "========================================="
echo "테스트 완료"
echo "========================================="
