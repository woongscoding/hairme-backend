# HairMe Backend - AI 헤어스타일 추천 서비스

[![Version](https://img.shields.io/badge/version-20.2.0-blue.svg)](https://github.com/woongscoding/hairme-backend)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20DynamoDB-orange.svg)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **MediaPipe, Google Gemini, AWS 서버리스 아키텍처를 활용한 AI 기반 헤어스타일 추천 서비스**

---

## 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [주요 기능](#주요-기능)
- [빠른 시작](#빠른-시작)
- [로컬 개발](#로컬-개발)
- [배포](#배포)
- [API 문서](#api-문서)
- [데이터베이스 스키마](#데이터베이스-스키마)
- [문제 해결](#문제-해결)

---

## 개요

HairMe Backend는 **서버리스** AI 기반 헤어스타일 추천 서비스입니다. **MediaPipe**를 사용해 얼굴 특징을 분석하고, **Google Gemini AI**를 통해 개인화된 헤어스타일을 추천합니다.

### 핵심 기술

| 기술 | 설명 |
|------|------|
| **FastAPI** | 모던 Python 웹 프레임워크 |
| **MediaPipe** | 실시간 얼굴 메시 분석 (478개 랜드마크) |
| **Google Gemini** | AI 기반 헤어스타일 추천 |
| **AWS Lambda** | 서버리스 컴퓨팅 |
| **DynamoDB** | NoSQL 데이터베이스 (온디맨드 과금) |
| **ECR** | Docker 컨테이너 레지스트리 |
| **Mangum** | AWS Lambda용 ASGI 어댑터 |

### 비용 및 성능

| 항목 | 수치 |
|------|------|
| 월 비용 | ~$0.00 (AWS 프리 티어) |
| 응답 시간 | 1.8s - 3.2s |
| 확장성 | 자동 스케일링 (1000+ 동시 요청) |
| 가용성 | 99.99% (AWS SLA) |

---

## 아키텍처

### 서버리스 아키텍처

```
┌──────────────┐
│ Android App  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│  Lambda Function │ ← FastAPI + Mangum
│  (Auto-scaling)  │
└──────┬───────────┘
       │
   ┌───┴────┬──────────┐
   ▼        ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐
│DynamoDB│ │Gemini  │ │  ECR   │
│        │ │  API   │ │ Images │
└────────┘ └────────┘ └────────┘
```

### 요청 흐름

1. **클라이언트**가 Lambda Function URL로 이미지 전송
2. **Lambda**가 FastAPI를 통해 요청 처리
3. **MediaPipe**가 얼굴 랜드마크 분석
4. **Gemini API**가 개인화된 추천 생성
5. **DynamoDB**에 분석 결과 저장
6. `analysis_id`와 함께 **응답** 반환

---

## 주요 기능

### 핵심 기능

- **얼굴 분석** - MediaPipe 478포인트 얼굴 메시
- **AI 추천** - Google Gemini 1.5 Flash
- **피드백 시스템** - 좋아요/싫어요 추적
- **검색 연동** - 네이버 이미지 검색 링크
- **분석 통계** - 피드백 통계 및 인사이트
- **캐싱** - Redis 호환 인메모리 캐시
- **로깅** - 구조화된 JSON 로깅

### API 엔드포인트

| 메서드 | 엔드포인트 | 설명 |
|--------|----------|------|
| `GET` | `/` | 서비스 상태 및 버전 |
| `GET` | `/api/health` | 헬스 체크 |
| `POST` | `/api/analyze` | 얼굴 분석 및 추천 |
| `POST` | `/api/v2/analyze-hybrid` | 하이브리드 분석 (MediaPipe + Gemini) |
| `POST` | `/api/feedback` | 사용자 피드백 제출 |
| `GET` | `/api/stats/feedback` | 피드백 통계 조회 |

### 서비스 아키텍처

#### 필수 서비스 (서버 시작 필수)

| 서비스 | 설명 | 실패 시 동작 |
|--------|------|-------------|
| **MediaPipe** | 478포인트 얼굴 메시 분석 | `RuntimeError` - 서버 시작 불가 |
| **Gemini API** | AI 추천 엔진 | `RuntimeError` - 서버 시작 불가 |
| **Hybrid Service** | Gemini + ML 추천 엔진 | `RuntimeError` - 서버 시작 불가 |

**Circuit Breaker 보호:**
- Gemini API 호출은 Circuit Breaker 패턴으로 보호
- 5회 연속 실패 → Circuit OPEN (60초 타임아웃)
- 폴백: 장애 시 MediaPipe 단독 분석

#### 선택 서비스

| 서비스 | 설명 | 실패 시 동작 |
|--------|------|-------------|
| **ML Model** | PyTorch 추천 모델 | 경고 로그 - 기본 점수 사용 |
| **Sentence Transformer** | 스타일 임베딩 | 경고 로그 - 임베딩 없이 진행 |
| **Feedback Collector** | 사용자 피드백 저장 | 경고 로그 - 피드백 비활성화 |
| **Retrain Queue** | 모델 재학습 큐 | 경고 로그 - 재학습 비활성화 |

---

## 빠른 시작

### 사전 요구사항

- Python 3.11+
- Docker & Docker Desktop
- AWS CLI 설정 완료
- Google Gemini API 키

### 1. 레포지토리 클론

```bash
git clone https://github.com/woongscoding/hairme-backend.git
cd hairme-backend
```

### 2. 환경 설정

```bash
# 환경 변수 템플릿 복사
cp .env.example .env

# .env 파일 편집
nano .env
```

**필수 환경 변수:**
```bash
# Gemini API
GEMINI_API_KEY=your_gemini_api_key_here

# Database (DynamoDB)
USE_DYNAMODB=true
AWS_REGION=ap-northeast-2
DYNAMODB_TABLE_NAME=hairme-analysis

# Application
APP_TITLE="HairMe API"
APP_VERSION="20.2.0"
MODEL_NAME=gemini-1.5-flash-latest
LOG_LEVEL=INFO
```

### 3. 의존성 설치

```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 4. 로컬 실행

```bash
# 개발 서버 시작
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API 테스트:**
```bash
# 헬스 체크
curl http://localhost:8000/api/health

# 이미지 분석
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@test_image.jpg"
```

---

## 로컬 개발

### 개발 환경 설정

#### 옵션 1: 로컬 Python

```bash
# 가상환경 활성화
source venv/bin/activate

# 개발 의존성 설치
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# 핫 리로드로 실행
uvicorn main:app --reload --log-level debug
```

#### 옵션 2: Docker

```bash
# Docker 이미지 빌드
docker build -t hairme-backend:dev .

# 컨테이너 실행
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e USE_DYNAMODB=false \
  hairme-backend:dev
```

#### 옵션 3: Lambda 로컬 테스트

```bash
# Lambda 컨테이너 빌드
docker build -f Dockerfile.lambda -t hairme-lambda:local .

# Lambda 런타임 로컬 실행
./scripts/test_lambda_local.sh --verbose
```

### 테스트 실행

```bash
# 전체 테스트
pytest tests/ -v

# 커버리지 포함
pytest tests/ --cov=. --cov-report=html

# 특정 테스트 파일
pytest tests/test_dynamodb_integration.py -v
```

### 코드 품질

```bash
# 코드 포맷팅
black .

# 린트
flake8 .

# 타입 체크
mypy .
```

---

## 배포

### AWS Lambda 배포

#### 자동 배포 (권장)

```bash
# 단일 명령으로 배포
./scripts/deploy_lambda.sh
```

**수행 작업:**
1. 사전 요구사항 검증 (AWS CLI, Docker)
2. 필요시 ECR 레포지토리 생성
3. 이전 Lambda 버전 백업
4. Docker 이미지 빌드 (linux/amd64)
5. ECR에 푸시
6. Lambda 함수 코드 및 설정 업데이트
7. 환경 변수 설정

#### 수동 배포

```bash
# 1. Lambda용 Docker 이미지 빌드
docker build -f Dockerfile.lambda \
  -t hairme-lambda:latest \
  --platform linux/amd64 .

# 2. ECR에 태그 및 푸시
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com

docker tag hairme-lambda:latest \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest

docker push <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest

# 3. Lambda 함수 업데이트
aws lambda update-function-code \
  --function-name hairme-analyze \
  --image-uri <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest \
  --region ap-northeast-2
```

### 배포 확인

```bash
# Lambda 함수 상태 확인
aws lambda get-function \
  --function-name hairme-analyze \
  --region ap-northeast-2

# Lambda 함수 테스트
aws lambda invoke \
  --function-name hairme-analyze \
  --payload '{"resource":"/api/health","path":"/api/health","httpMethod":"GET"}' \
  output.json

# 로그 모니터링
aws logs tail /aws/lambda/hairme-analyze --follow
```

---

## API 문서

### POST /api/analyze

얼굴 특징을 분석하고 개인화된 헤어스타일을 추천합니다.

**요청:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@face_image.jpg"
```

**응답:**
```json
{
  "success": true,
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "data": {
    "face_shape": "oval",
    "gemini_shape": "타원형",
    "recommended_styles": [
      {
        "style_name": "레이어드 컷",
        "reason": "타원형 얼굴은 대부분의 스타일이 잘 어울립니다",
        "image_search_url": "https://search.naver.com/..."
      }
    ],
    "mediapipe_features": {
      "face_ratio": 1.35,
      "forehead_ratio": 0.45,
      "cheekbone_ratio": 0.89,
      "jawline_ratio": 0.72
    }
  },
  "processing_time": 2.45
}
```

### POST /api/feedback

특정 헤어스타일 추천에 대한 사용자 피드백을 제출합니다.

**요청:**
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "style_index": 1,
    "feedback": "good",
    "naver_clicked": true
  }'
```

**응답:**
```json
{
  "success": true,
  "message": "피드백이 성공적으로 저장되었습니다"
}
```

### GET /api/stats/feedback

집계된 피드백 통계를 조회합니다.

**요청:**
```bash
curl http://localhost:8000/api/stats/feedback
```

**응답:**
```json
{
  "total_feedbacks": 1523,
  "good_count": 987,
  "bad_count": 536,
  "good_rate": 64.8,
  "styles": {
    "레이어드 컷": {"good": 245, "bad": 89},
    "단발": {"good": 198, "bad": 112}
  },
  "naver_click_rate": 45.2
}
```

---

## 데이터베이스 스키마

### DynamoDB 테이블: `hairme-analysis`

**Primary Key:**
- `analysis_id` (String, UUID) - Partition key

**Global Secondary Index:**
- `created_at-index` - 생성 시간별 정렬
  - Partition key: `entity_type` (String, 항상 "ANALYSIS")
  - Sort key: `created_at` (String, ISO 8601)

**주요 속성:**

| 속성 | 타입 | 설명 |
|------|------|------|
| `analysis_id` | String | UUID 기본 키 |
| `created_at` | String | ISO 8601 타임스탬프 |
| `image_hash` | String | 이미지 SHA-256 해시 |
| `face_shape` | String | 감지된 얼굴형 |
| `gemini_shape` | String | Gemini의 얼굴형 해석 |
| `recommended_styles` | List | 스타일 객체 배열 |
| `style_1_feedback` | String | "good" / "bad" / null |
| `style_1_naver_clicked` | Boolean | 검색 링크 클릭 여부 |
| `mediapipe_*` | Number | 얼굴 측정값 |
| `processing_time` | Number | 분석 소요 시간 (초) |

---

## 문제 해결

### 자주 발생하는 문제

#### 1. Lambda 콜드 스타트가 느림

**증상:** 첫 번째 요청이 5-10초 소요

**해결:**
```bash
# 옵션 1: Provisioned Concurrency 사용 (비용 발생)
aws lambda put-provisioned-concurrency-config \
  --function-name hairme-analyze \
  --provisioned-concurrent-executions 1

# 옵션 2: 콜드 스타트 허용 (대부분의 요청은 웜 상태)
# 15분 비활성 후에만 콜드 스타트 발생
```

#### 2. DynamoDB 연결 오류

**증상:** `DynamoDB not initialized`

**해결:**
```bash
# 1. 환경 변수 확인
grep USE_DYNAMODB .env

# 2. AWS 자격 증명 확인
aws sts get-caller-identity

# 3. 테이블 존재 확인
aws dynamodb describe-table \
  --table-name hairme-analysis \
  --region ap-northeast-2
```

#### 3. Gemini API 오류

**증상:** `Gemini API call failed`

**해결:**
```bash
# 1. API 키 설정 확인
echo $GEMINI_API_KEY

# 2. API 키 테스트
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=$GEMINI_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'
```

#### 4. MediaPipe Import 오류

**증상:** `ImportError: libGL.so.1: cannot open shared object file`

**해결:**
`Dockerfile.lambda`에 이미 수정됨:
```dockerfile
RUN yum install -y mesa-libGL
```

### 디버그 모드

```bash
# 디버그 로깅 활성화
export LOG_LEVEL=DEBUG

# 상세 출력으로 실행
python -m uvicorn main:app --log-level debug

# CloudWatch 로그 확인 (Lambda)
aws logs tail /aws/lambda/hairme-analyze --follow --filter-pattern "ERROR"
```

---

## 프로젝트 상태

**현재 버전:** 20.2.0 (DynamoDB + Lambda)
**아키텍처:** 서버리스 (AWS Lambda + DynamoDB)
**상태:** Production Ready
**월 비용:** ~$0.00 (프리 티어)

### 최근 업데이트

- RDS MySQL에서 DynamoDB로 마이그레이션 (100% 데이터 이관)
- 서버리스 아키텍처 배포 (Lambda + DynamoDB)
- 비용 95% 절감 ($61/월 → $0/월)
- 쿼리 성능 20-75% 향상
- 종합 테스트 스위트 구현

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 개발자 정보

**개발자:** 박찬웅
**이메일:** [mapcw99lol@gmail.com](mailto:mapcw99lol@gmail.com)
**GitHub:** [woongscoding/hairme-backend](https://github.com/woongscoding/hairme-backend)

---

## 감사의 글

- **Google Gemini** - AI 기반 추천
- **MediaPipe** - 얼굴 메시 분석
- **FastAPI** - 모던 Python 웹 프레임워크
- **AWS** - 서버리스 인프라
