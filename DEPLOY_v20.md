# HairMe v20 배포 가이드 (피드백 기능)

## 📋 변경 사항 요약

### 1. DB 스키마 변경
- `recommended_styles` (JSON): 추천된 3개 헤어스타일 저장
- `style_1/2/3_feedback` (ENUM): 각 스타일별 좋아요/싫어요
- `style_1/2/3_naver_clicked` (BOOLEAN): 네이버 검색 클릭 여부
- `feedback_at` (DATETIME): 피드백 제출 시각

### 2. API 변경
- `/api/analyze`: 응답에 `analysis_id` 추가
- `/api/feedback`: 새로운 엔드포인트 추가

### 3. 버전 업데이트
- v19 → v20

---

## 🚀 배포 단계

### Step 1: DB 스키마 업데이트

RDS에 접속해서 SQL 실행:

```bash
# 로컬에서 RDS 접속
mysql -h hairme-data.cr28a6uqo2k8.ap-northeast-2.rds.amazonaws.com \
      -u admin -p hairme < db_schema_v20.sql
```

또는 AWS Systems Manager Session Manager를 통해 접속.

### Step 2: 로컬 테스트 (선택사항)

```bash
# Docker 빌드
docker build -t hairstyle-api:v20 .

# 로컬 실행 (환경변수 필요)
docker run -p 8000:8000 \
  -e DATABASE_URL="mysql+asyncmy://admin@localhost:3306/hairme" \
  -e DB_PASSWORD="your_password" \
  -e GEMINI_API_KEY="your_api_key" \
  hairstyle-api:v20

# 테스트
curl http://localhost:8000/
curl http://localhost:8000/api/health
```

### Step 3: ECR 푸시

```bash
# AWS 로그인
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  364042451408.dkr.ecr.ap-northeast-2.amazonaws.com

# 이미지 태그
docker tag hairstyle-api:v20 \
  364042451408.dkr.ecr.ap-northeast-2.amazonaws.com/hairstyle-api:v20

# 푸시
docker push 364042451408.dkr.ecr.ap-northeast-2.amazonaws.com/hairstyle-api:v20
```

### Step 4: ECS 업데이트

```bash
# Task Definition 등록
aws ecs register-task-definition \
  --cli-input-json file://task-def-v20.json \
  --region ap-northeast-2

# 서비스 업데이트 (기존 서비스 이름 확인 필요)
aws ecs update-service \
  --cluster hairme-cluster \
  --service hairme-service \
  --task-definition hairme-task \
  --force-new-deployment \
  --region ap-northeast-2
```

### Step 5: 배포 확인

```bash
# ECS 서비스 상태 확인
aws ecs describe-services \
  --cluster hairme-cluster \
  --services hairme-service \
  --region ap-northeast-2

# ALB를 통해 API 테스트
curl https://your-alb-url.ap-northeast-2.elb.amazonaws.com/
curl https://your-alb-url.ap-northeast-2.elb.amazonaws.com/api/health
```

---

## 🧪 API 테스트

### 1. 얼굴 분석 (기존 + analysis_id 추가)

```bash
curl -X POST https://your-alb-url/api/analyze \
  -F "file=@test_face.jpg"

# 응답 예시
{
  "success": true,
  "analysis_id": 123,  # ✅ 새로 추가된 필드
  "data": {
    "analysis": {
      "face_shape": "계란형",
      "personal_color": "봄웜",
      "features": "..."
    },
    "recommendations": [
      {
        "style_name": "레이어드 컷",
        "reason": "...",
        "image_search_url": "https://search.naver.com/..."
      }
    ]
  }
}
```

### 2. 피드백 제출 (신규)

```bash
curl -X POST https://your-alb-url/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": 123,
    "style_index": 1,
    "feedback": "like",
    "naver_clicked": true
  }'

# 응답 예시
{
  "success": true,
  "message": "피드백이 저장되었습니다",
  "analysis_id": 123,
  "style_index": 1
}
```

---

## 📊 CloudWatch 로그 확인

```bash
# 로그 스트림 확인
aws logs tail /ecs/hairstyle-api --follow --region ap-northeast-2

# 피드백 이벤트만 필터링
aws logs filter-log-events \
  --log-group-name /ecs/hairstyle-api \
  --filter-pattern '{ $.event_type = "feedback_submitted" }' \
  --region ap-northeast-2
```

---

## 🔄 롤백 방법

문제 발생 시 v19로 롤백:

```bash
aws ecs update-service \
  --cluster hairme-cluster \
  --service hairme-service \
  --task-definition hairme-task:이전버전번호 \
  --force-new-deployment \
  --region ap-northeast-2
```

---

## ✅ 배포 후 체크리스트

- [ ] `/api/health` 응답에 `"feedback_system": "enabled"` 확인
- [ ] 얼굴 분석 응답에 `analysis_id` 포함 확인
- [ ] 피드백 API 호출 성공 확인
- [ ] DB에 피드백 데이터 저장 확인
- [ ] CloudWatch에 `feedback_submitted` 이벤트 로그 확인

---

## 📝 다음 단계

v20 배포 후:
1. 안드로이드 앱 UI 수정 (좋아요/싫어요 버튼)
2. 피드백 API 호출 로직 구현
3. 개인정보처리방침 작성
4. 플레이스토어 배포 준비
