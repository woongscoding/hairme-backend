# BeautyMe Backend - AI Beauty Consulting Platform

[![Version](https://img.shields.io/badge/version-23.0.0-blue.svg)](https://github.com/woongscoding/hairme-backend)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20DynamoDB-orange.svg)](https://aws.amazon.com/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **얼굴 분석, 퍼스널컬러 진단, AI 헤어스타일 추천 및 합성을 제공하는 종합 뷰티 컨설팅 플랫폼**

---

## 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [주요 기능](#주요-기능)
- [API 엔드포인트](#api-엔드포인트)
- [서비스 아키텍처](#서비스-아키텍처)
- [빠른 시작](#빠른-시작)
- [로컬 개발](#로컬-개발)
- [배포](#배포)
- [데이터베이스 스키마](#데이터베이스-스키마)
- [환경 변수](#환경-변수)
- [보안](#보안)
- [문제 해결](#문제-해결)

---

## 개요

BeautyMe Backend는 **서버리스** AI 기반 종합 뷰티 컨설팅 플랫폼입니다.
**MediaPipe**로 얼굴 특징을 분석하고, **ITA 알고리즘**으로 퍼스널컬러를 진단하며, **PyTorch ML 모델**로 개인화된 헤어스타일을 추천하고, **Gemini 2.5 Flash Image**로 헤어스타일/헤어컬러를 합성합니다.

### 핵심 기술

| 기술 | 설명 |
|------|------|
| **FastAPI** | 모던 Python 웹 프레임워크 |
| **MediaPipe** | 실시간 얼굴 메시 분석 (478개 랜드마크) |
| **Google Gemini 2.5 Flash** | AI 기반 얼굴 분석 및 뷰티 컨설팅 |
| **Gemini 2.5 Flash Image** | AI 헤어스타일/헤어컬러 합성 (google-genai SDK) |
| **PyTorch** | ML 기반 헤어스타일 추천 모델 |
| **Sentence Transformers** | 스타일 임베딩 생성 (paraphrase-multilingual-MiniLM) |
| **AWS Lambda** | 서버리스 컴퓨팅 |
| **DynamoDB** | NoSQL 데이터베이스 (온디맨드 과금) |
| **ECR** | Docker 컨테이너 레지스트리 |
| **Mangum** | AWS Lambda용 ASGI 어댑터 |
| **SlowAPI** | 엔드포인트별 Rate Limiting |
| **pybreaker** | Circuit Breaker 패턴 (Gemini API 보호) |
| **Sentry** | 에러 트래킹 및 성능 모니터링 |

### 비용 및 성능

| 항목 | 수치 |
|------|------|
| 월 비용 | Lambda 기반 최소 비용 구조 (요청 당 과금, 유휴 시 $0) |
| 응답 시간 | 1.8s - 3.2s (분석), 3-8s (합성) |
| 확장성 | 자동 스케일링 (1000+ 동시 요청) |
| 가용성 | 99.99% (AWS SLA) |

---

## 아키텍처

### 서버리스 아키텍처

```
┌──────────────────┐
│   Android App    │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│                Lambda Function (FastAPI + Mangum)       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Rate Limit  │  │Security Hdrs │  │  File Size    │ │
│  │ (SlowAPI)   │  │ Middleware   │  │  Middleware    │ │
│  └─────────────┘  └──────────────┘  └───────────────┘ │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │              API Routers                        │   │
│  │  Beauty │ PersonalColor │ HairColor │ Synthesis │   │
│  │  Analyze│ Feedback │ Usage │ Admin              │   │
│  └─────────────────────────────────────────────────┘   │
│                                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Services                           │   │
│  │  FaceDetection  │ GeminiAnalysis │ PersonalColor│   │
│  │  HairColor │ HairstyleSynthesis │ BeautyConsult│   │
│  │  HybridRecommender │ UsageLimit │ TrendingStyle│   │
│  │  CircuitBreaker │ MLOps │ A/B Testing          │   │
│  └─────────────────────────────────────────────────┘   │
└──────────┬─────────┬──────────┬──────────┬─────────────┘
           │         │          │          │
     ┌─────┘    ┌────┘     ┌───┘     ┌────┘
     ▼          ▼          ▼         ▼
┌─────────┐ ┌────────┐ ┌────────┐ ┌─────────┐
│DynamoDB │ │Gemini  │ │  ECR   │ │Secrets  │
│(2 table)│ │  API   │ │ Images │ │Manager  │
└─────────┘ └────────┘ └────────┘ └─────────┘
  analysis     2.5 Flash            API Keys
  usage        Image API
```

### 요청 흐름

1. **클라이언트**가 Lambda Function URL로 이미지 전송
2. **미들웨어**가 Rate Limit, Security Headers, 파일 크기 검증
3. **MediaPipe**가 얼굴 랜드마크 분석 (478포인트)
4. **ITA 알고리즘**이 퍼스널컬러 진단
5. **PyTorch ML 모델**이 개인화된 헤어스타일 추천
6. **Gemini 2.5 Flash Image**로 헤어스타일/헤어컬러 합성 (선택)
7. **DynamoDB**에 분석 결과 저장
8. 종합 뷰티 리포트 **응답** 반환

---

## 주요 기능

- **얼굴 분석** - MediaPipe 478포인트 얼굴 메시 + Gemini Vision 하이브리드 분석
- **퍼스널컬러 진단** - ITA 기반 피부톤 분석 (봄웜/여름쿨/가을웜/겨울쿨)
- **AI 헤어스타일 추천** - PyTorch ML 모델 + Sentence Transformer 임베딩
- **헤어스타일 합성** - Gemini 2.5 Flash Image로 헤어스타일 적용 미리보기
- **헤어컬러 추천/합성** - 퍼스널컬러 기반 추천 + 가상 염색 시뮬레이션
- **뷰티 컨설팅** - 종합 분석 리포트 + AI 챗봇 상담
- **피드백 시스템** - 좋아요/싫어요 추적 + MLOps 재학습 파이프라인
- **사용량 관리** - 디바이스별 일일 합성 횟수 제한
- **A/B 테스팅** - Champion/Challenger 모델 실험
- **Circuit Breaker** - Gemini API 장애 시 자동 폴백

---

## API 엔드포인트

### 시스템

| 메서드 | 엔드포인트 | 설명 |
|--------|----------|------|
| `GET` | `/` | 서비스 상태, 버전, 사용 가능한 기능 목록 |
| `GET` | `/api/health` | 헬스 체크 (`?deep=true`로 상세 점검) |

### 뷰티 컨설팅

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `POST` | `/api/beauty/analyze` | 종합 뷰티 분석 (얼굴형 + 퍼스널컬러 + 추천) | 10/min |
| `POST` | `/api/beauty/consult` | AI 뷰티 상담 챗봇 | 30/min |
| `POST` | `/api/beauty/report` | 종합 뷰티 분석 리포트 생성 (markdown/json) | 5/min |
| `GET` | `/api/beauty/features` | 플랫폼 기능 및 capabilities 조회 | - |

### 퍼스널컬러

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `POST` | `/api/personal-color` | 퍼스널컬러 분석 (ITA 기반 피부톤 진단) | 10/min |
| `GET` | `/api/personal-color/{color_type}/palette` | 퍼스널컬러 타입별 컬러 팔레트 | - |
| `GET` | `/api/personal-color/{color_type}/styling` | 퍼스널컬러별 스타일링 팁 (메이크업/패션) | - |
| `GET` | `/api/personal-color/{color_type}/hair` | 퍼스널컬러별 추천 헤어 컬러 | - |
| `GET` | `/api/personal-color/types` | 지원되는 퍼스널컬러 타입 목록 | - |

### 헤어 컬러

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `GET` | `/api/hair-color/{personal_color}` | 퍼스널컬러 기반 헤어 컬러 추천 | - |
| `GET` | `/api/hair-color/trends/all` | 시즌별 트렌드 헤어 컬러 | - |
| `GET` | `/api/hair-color/search/{color_name}` | 헤어 컬러명 검색 | - |
| `POST` | `/api/hair-color/synthesize` | 가상 헤어 컬러 시뮬레이션 | 5/min |
| `POST` | `/api/hair-color/synthesize-by-personal-color` | 퍼스널컬러 기반 헤어 컬러 합성 | 5/min |

### 헤어스타일 합성

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `POST` | `/api/v2/synthesize` | Gemini 2.5 Flash Image 헤어스타일 합성 | 5/min |
| `POST` | `/api/v2/synthesize-with-reference` | 레퍼런스 이미지 기반 헤어스타일 합성 | 3/min |

### 얼굴 분석

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `POST` | `/api/analyze` | ML 기반 얼굴 분석 및 헤어스타일 추천 | 10/min |
| `POST` | `/api/v2/analyze-hybrid` | 하이브리드 분석 (MediaPipe + ML + Gemini) | 10/min |

### 피드백

| 메서드 | 엔드포인트 | 설명 | Rate Limit |
|--------|----------|------|-----------|
| `POST` | `/api/feedback` | 사용자 피드백 제출 | 20/min |
| `GET` | `/api/stats/feedback` | 피드백 통계 조회 | 30/min |

### 사용량

| 메서드 | 엔드포인트 | 설명 |
|--------|----------|------|
| `GET` | `/api/usage` | 디바이스별 일일 합성 잔여 횟수 조회 |
| `POST` | `/api/usage/consume` | 사용량 차감 (합성 엔드포인트에서 자동 처리) |

### 관리자 (Admin API Key 필요)

| 메서드 | 엔드포인트 | 설명 |
|--------|----------|------|
| `GET` | `/api/admin/mlops-status` | MLOps 파이프라인 상태 |
| `GET` | `/api/admin/feedback-stats` | DynamoDB 기반 피드백 통계 |
| `GET` | `/api/admin/circuit-breaker-status` | Circuit Breaker 상태 |
| `POST` | `/api/admin/circuit-breaker-reset` | Circuit Breaker 수동 리셋 |
| `GET` | `/api/admin/abtest/status` | A/B 테스트 현재 상태 |
| `GET` | `/api/admin/abtest/metrics/{experiment_id}` | A/B 테스트 메트릭 |
| `GET` | `/api/admin/abtest/summary/{experiment_id}` | A/B 테스트 요약 |
| `POST` | `/api/admin/abtest/start` | A/B 테스트 시작 |
| `POST` | `/api/admin/abtest/stop` | A/B 테스트 중지 |
| `POST` | `/api/admin/abtest/promote/{experiment_id}` | Challenger 모델 승격 |

---

## 서비스 아키텍처

### 필수 서비스 (서버 시작 필수)

| 서비스 | 설명 | 실패 시 동작 |
|--------|------|-------------|
| **FaceDetectionService** | 얼굴 감지 (MediaPipe → Gemini 폴백) | `RuntimeError` - 서버 시작 불가 |
| **GeminiAnalysisService** | Gemini Vision 기반 AI 분석 (Circuit Breaker 보호) | `RuntimeError` - 서버 시작 불가 |
| **HybridRecommender** | PyTorch ML + Gemini 하이브리드 추천 엔진 | `RuntimeError` - 서버 시작 불가 |

### 핵심 비즈니스 서비스

| 서비스 | 설명 | 실패 시 동작 |
|--------|------|-------------|
| **PersonalColorService** | ITA 기반 퍼스널컬러 진단 | 경고 로그 - 퍼스널컬러 비활성화 |
| **HairColorService** | 헤어 컬러 추천 + 트렌드 데이터 | 경고 로그 - 헤어컬러 비활성화 |
| **HairstyleSynthesisService** | Gemini 2.5 Flash Image 합성 (리트라이 3회) | 경고 로그 - 합성 비활성화 |
| **BeautyConsultantService** | 종합 뷰티 프로필 생성 (모든 서비스 통합) | 경고 로그 - 개별 서비스로 폴백 |
| **UsageLimitService** | DynamoDB 기반 디바이스별 일일 합성 제한 | 경고 로그 - 제한 없이 진행 |
| **TrendingStyleService** | 트렌드 헤어스타일 데이터 제공 | 경고 로그 - 트렌드 비활성화 |

### MLOps 서비스

| 서비스 | 설명 | 실패 시 동작 |
|--------|------|-------------|
| **ML Model (PyTorch)** | v6 헤어스타일 추천 모델 | 경고 로그 - 기본 점수 사용 |
| **Sentence Transformer** | 스타일 임베딩 생성 | 경고 로그 - 임베딩 없이 진행 |
| **ReasonGenerator** | 템플릿 기반 추천 이유 생성 | 경고 로그 - 기본 설명 사용 |
| **Feedback Collector** | 사용자 피드백 저장 + S3 적재 | 경고 로그 - 피드백 비활성화 |
| **A/B Test Manager** | Champion/Challenger 실험 관리 | 경고 로그 - Champion 모델만 사용 |

**Circuit Breaker 보호:**
- Gemini API 호출은 Circuit Breaker 패턴으로 보호 (pybreaker)
- 5회 연속 실패 → Circuit OPEN (60초 타임아웃)
- 상태 전환 시 로깅 (open/close/half-open)

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
GEMINI_API_KEY=your_gemini_api_key_here
USE_DYNAMODB=true
AWS_REGION=ap-northeast-2
DYNAMODB_TABLE_NAME=hairme-analysis
MODEL_NAME=gemini-2.5-flash
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
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**API 테스트:**
```bash
# 헬스 체크
curl http://localhost:8000/api/health

# 종합 뷰티 분석
curl -X POST http://localhost:8000/api/beauty/analyze \
  -F "file=@face_image.jpg"

# 퍼스널컬러 분석
curl -X POST http://localhost:8000/api/personal-color \
  -F "file=@face_image.jpg"
```

---

## 로컬 개발

### 개발 환경 설정

#### 옵션 1: 로컬 Python

```bash
source venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

uvicorn main:app --reload --log-level debug
```

#### 옵션 2: Docker

```bash
docker build -t beautyme-backend:dev .

docker run -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e USE_DYNAMODB=false \
  beautyme-backend:dev
```

#### 옵션 3: Lambda 로컬 테스트

```bash
docker build -f Dockerfile.lambda -t beautyme-lambda:local .
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

### GitHub Actions CI/CD

배포는 **GitHub Actions** (`deploy.yml`)를 통해 자동으로 수행됩니다.
`main` 브랜치에 push하면 CI/CD 파이프라인이 트리거됩니다.

#### 파이프라인 단계

```
Push to main
    │
    ▼
┌──────────────────────────┐
│  1. Test                 │  ← 모든 push/PR에서 실행
│  - Black 포맷팅 체크     │
│  - Flake8 린트           │
│  - MyPy 타입 체크        │
│  - Pytest + 커버리지     │
└──────────┬───────────────┘
           │ main 브랜치만
           ▼
┌──────────────────────────┐
│  2. Build & Deploy       │
│  - 현재 Lambda 버전 백업 │
│  - Docker 이미지 빌드    │
│    (linux/amd64)         │
│  - ECR 푸시              │
│    (hairme-lambda-proxy) │
│  - Lambda 함수 업데이트  │
│    (메모리: 2048MB,      │
│     타임아웃: 30s)       │
│  - 환경 변수 설정        │
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  3. Health Check         │
│  - /api/health 호출      │
│  - "healthy" 응답 확인   │
│  - 배포 완료 알림        │
└──────────────────────────┘
```

#### 수동 배포

```bash
# 1. Lambda용 Docker 이미지 빌드
docker build -f Dockerfile.lambda \
  -t hairme-lambda-proxy:latest \
  --platform linux/amd64 .

# 2. ECR에 태그 및 푸시
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com

docker tag hairme-lambda-proxy:latest \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda-proxy:latest

docker push \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda-proxy:latest

# 3. Lambda 함수 업데이트
aws lambda update-function-code \
  --function-name hairme-lambda-proxy \
  --image-uri <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda-proxy:latest \
  --region ap-northeast-2
```

### 배포 확인

```bash
# Lambda 함수 상태 확인
aws lambda get-function \
  --function-name hairme-lambda-proxy \
  --region ap-northeast-2

# 헬스 체크
curl https://<function-url>/api/health

# 로그 모니터링
aws logs tail /aws/lambda/hairme-lambda-proxy --follow
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
| `personal_color` | String | 퍼스널컬러 진단 결과 |
| `recommended_styles` | List | 스타일 객체 배열 |
| `style_1_feedback` | String | "good" / "bad" / null |
| `style_1_naver_clicked` | Boolean | 검색 링크 클릭 여부 |
| `mediapipe_*` | Number | 얼굴 측정값 |
| `processing_time` | Number | 분석 소요 시간 (초) |

### DynamoDB 테이블: `hairstyle_usage`

**Primary Key:**
- `device_id` (String) - Partition key
- `date` (String, YYYY-MM-DD KST) - Sort key

**주요 속성:**

| 속성 | 타입 | 설명 |
|------|------|------|
| `device_id` | String | 디바이스 고유 식별자 |
| `date` | String | 날짜 (KST 기준) |
| `count` | Number | 당일 합성 사용 횟수 |
| `ttl` | Number | 자동 삭제 시간 (7일 후, epoch) |

---

## 환경 변수

### 필수

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `GEMINI_API_KEY` | Google Gemini API 키 | - (필수) |
| `USE_DYNAMODB` | DynamoDB 사용 여부 | `false` |
| `AWS_REGION` | AWS 리전 | `ap-northeast-2` |
| `DYNAMODB_TABLE_NAME` | 분석 결과 테이블명 | `hairme-analysis` |
| `MODEL_NAME` | Gemini 모델명 | `gemini-2.5-flash` |

### 사용량 관리

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `DYNAMODB_USAGE_TABLE_NAME` | 사용량 테이블명 | `hairstyle_usage` |
| `DAILY_SYNTHESIS_LIMIT` | 디바이스별 일일 합성 제한 | `3` |

### 관리자

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `ADMIN_API_KEY` | 관리자 엔드포인트 인증 키 | `None` |

### MLOps

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `MLOPS_ENABLED` | MLOps 파이프라인 활성화 | `false` |
| `MLOPS_S3_BUCKET` | MLOps S3 버킷 | `hairme-mlops` |
| `MLOPS_RETRAIN_THRESHOLD` | 재학습 트리거 피드백 수 | `100` |
| `MLOPS_TRAINER_LAMBDA` | Trainer Lambda 함수명 | `hairme-model-trainer` |

### A/B 테스트

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `ABTEST_ENABLED` | A/B 테스트 활성화 | `false` |
| `ABTEST_EXPERIMENT_ID` | 실험 ID | `""` |
| `ABTEST_CHAMPION_VERSION` | Champion 모델 버전 | `v6` |
| `ABTEST_CHALLENGER_VERSION` | Challenger 모델 버전 | `""` |
| `ABTEST_CHALLENGER_PERCENT` | Challenger 트래픽 비율 (%) | `10` |

### 모니터링

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `SENTRY_DSN` | Sentry DSN (에러 트래킹) | `None` |
| `LOG_LEVEL` | 로그 레벨 | `INFO` |
| `ENVIRONMENT` | 환경 (development/production) | `development` |

### 기타

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `ALLOWED_ORIGINS` | CORS 허용 출처 (쉼표 구분) | `http://localhost:3000` |
| `REDIS_URL` | Redis 캐시 URL | `None` |
| `CACHE_TTL` | 캐시 TTL (초) | `86400` |

---

## 보안

### 인증 및 접근 제어

- **Admin API Key** - 관리자 엔드포인트 접근 시 API Key 헤더 필수
- **AWS Secrets Manager** - 프로덕션 환경에서 API 키 안전 관리 (GEMINI_API_KEY, ADMIN_API_KEY)

### Rate Limiting (SlowAPI)

| 엔드포인트 | 제한 |
|-----------|------|
| 합성 (synthesize) | 5 req/min |
| 레퍼런스 합성 | 3 req/min |
| 분석 (analyze) | 10 req/min |
| 피드백 제출 | 20 req/min |
| 뷰티 상담 | 30 req/min |

### Security Headers 미들웨어

| 헤더 | 설정 |
|------|------|
| Content-Security-Policy | strict (self + data/https for images) |
| X-Frame-Options | DENY |
| X-Content-Type-Options | nosniff |
| X-XSS-Protection | 1; mode=block |
| Strict-Transport-Security | max-age=31536000 (HTTPS 환경) |
| Referrer-Policy | strict-origin-when-cross-origin |
| Permissions-Policy | geolocation, microphone, camera, payment, usb 차단 |

### 파일 업로드 제한

- 최대 파일 크기: **10MB**
- 허용 포맷: JPG, PNG, WEBP
- File Size Middleware로 DoS 방지

---

## 문제 해결

### 자주 발생하는 문제

#### 1. Lambda 콜드 스타트가 느림

**증상:** 첫 번째 요청이 5-10초 소요

**해결:**
```bash
# 옵션 1: Provisioned Concurrency 사용 (비용 발생)
aws lambda put-provisioned-concurrency-config \
  --function-name hairme-lambda-proxy \
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

**증상:** `Gemini API call failed` 또는 Circuit Breaker OPEN

**해결:**
```bash
# 1. API 키 설정 확인
echo $GEMINI_API_KEY

# 2. Circuit Breaker 상태 확인 (Admin API)
curl -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  http://localhost:8000/api/admin/circuit-breaker-status

# 3. Circuit Breaker 수동 리셋
curl -X POST -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  http://localhost:8000/api/admin/circuit-breaker-reset
```

#### 4. MediaPipe Import 오류

**증상:** `ImportError: libGL.so.1: cannot open shared object file`

**해결:**
`Dockerfile.lambda`에 이미 포함:
```dockerfile
RUN yum install -y mesa-libGL
```

#### 5. 합성 횟수 제한 초과

**증상:** `Daily synthesis limit exceeded`

**해결:**
- 기본 일일 합성 제한: 3회/디바이스
- `DAILY_SYNTHESIS_LIMIT` 환경 변수로 조정 가능
- DynamoDB `hairstyle_usage` 테이블에서 사용량 확인

### 디버그 모드

```bash
# 디버그 로깅 활성화
export LOG_LEVEL=DEBUG

# 상세 출력으로 실행
python -m uvicorn main:app --log-level debug

# CloudWatch 로그 확인 (Lambda)
aws logs tail /aws/lambda/hairme-lambda-proxy --follow --filter-pattern "ERROR"
```

---

## 프로젝트 상태

**현재 버전:** 23.0.0 (BeautyMe - 종합 뷰티 컨설팅)
**아키텍처:** 서버리스 (AWS Lambda + DynamoDB)
**배포:** GitHub Actions CI/CD → ECR → Lambda
**상태:** Production Ready

### 주요 마일스톤

- **v20.x** - DynamoDB 마이그레이션, 서버리스 전환
- **v21.x** - 퍼스널컬러 진단, 헤어컬러 추천 추가
- **v22.x** - Gemini 2.5 Flash Image 합성, 사용량 제한
- **v23.0** - 종합 뷰티 컨설팅, A/B 테스팅, MLOps 파이프라인, GitHub Actions CI/CD

---

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 개발자 정보

**개발자:** 박찬웅
**이메일:** [mapcw99lol@gmail.com](mailto:mapcw99lol@gmail.com)
**GitHub:** [woongscoding/hairme-backend](https://github.com/woongscoding/hairme-backend)
