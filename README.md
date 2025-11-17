# HairMe Backend - AI-Powered Hairstyle Recommendation

[![Version](https://img.shields.io/badge/version-20.2.0-blue.svg)](https://github.com/your-repo/hairme-backend)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![AWS](https://img.shields.io/badge/AWS-Lambda%20%2B%20DynamoDB-orange.svg)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **AI-powered hairstyle recommendation service using MediaPipe, Google Gemini, and AWS serverless architecture**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Quick Start](#quick-start)
- [Local Development](#local-development)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Database Schema](#database-schema)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎯 Overview

HairMe Backend is a **serverless** AI-powered hairstyle recommendation service that analyzes facial features using **MediaPipe** and provides personalized hairstyle recommendations using **Google Gemini AI**.

### Key Technologies

- **FastAPI** - Modern Python web framework
- **MediaPipe** - Real-time face mesh analysis (478 landmarks)
- **Google Gemini** - AI-powered hairstyle recommendations
- **AWS Lambda** - Serverless compute
- **DynamoDB** - NoSQL database with on-demand pricing
- **ECR** - Docker container registry
- **Mangum** - ASGI adapter for AWS Lambda

### Cost & Performance

- **Monthly Cost:** ~$0.00 (within AWS Free Tier)
- **Response Time:** 1.8s - 3.2s (analysis)
- **Scalability:** Auto-scaling (1000+ concurrent requests)
- **Availability:** 99.99% (AWS SLA)

---

## 🏗️ Architecture

### Current Architecture (Serverless)

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

### Request Flow

1. **Client** sends image to Lambda function URL
2. **Lambda** processes request via FastAPI
3. **MediaPipe** analyzes facial landmarks
4. **Gemini API** generates personalized recommendations
5. **DynamoDB** stores analysis results
6. **Response** returned to client with `analysis_id`

---

## ✨ Features

### Core Features

- ✅ **Face Analysis** - MediaPipe 478-point face mesh
- ✅ **AI Recommendations** - Google Gemini 1.5 Flash
- ✅ **Feedback System** - Like/dislike tracking
- ✅ **Search Integration** - Naver image search links
- ✅ **Analytics** - Feedback statistics and insights
- ✅ **Caching** - Redis-compatible in-memory cache
- ✅ **Logging** - Structured JSON logging

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service status and version |
| `GET` | `/api/health` | Health check with service status |
| `POST` | `/api/analyze` | Analyze face and get recommendations |
| `POST` | `/api/v2/analyze-hybrid` | Hybrid analysis (MediaPipe + Gemini) |
| `POST` | `/api/feedback` | Submit user feedback |
| `GET` | `/api/stats/feedback` | Get feedback statistics |

### Service Architecture & Dependencies

#### Required Services (서버 시작 필수)

These services must be initialized successfully, or the server will fail to start:

| Service | Description | Failure Behavior |
|---------|-------------|------------------|
| **MediaPipe** | 478-point face mesh analysis | `RuntimeError` - Server won't start |
| **Gemini API** | AI-powered recommendations | `RuntimeError` - Server won't start |
| **Hybrid Service** | Gemini + ML recommendation engine | `RuntimeError` - Server won't start |

**Circuit Breaker Protection:**
- Gemini API calls are protected by Circuit Breaker pattern
- 5 consecutive failures → Circuit OPEN (60s timeout)
- Fallback: MediaPipe-only analysis during outages

#### Optional Services (서버 시작 가능)

These services are optional and won't prevent server startup if they fail:

| Service | Description | Failure Behavior |
|---------|-------------|------------------|
| **ML Model** | PyTorch recommender model | Warning logged - Uses default scores |
| **Sentence Transformer** | Style embedding for similarity | Warning logged - Proceeds without embedding |
| **Feedback Collector** | User feedback storage | Warning logged - Feedback disabled |
| **Retrain Queue** | Model retraining queue | Warning logged - Retraining disabled |

**Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "20.2.0",
  "services": {
    "mediapipe": true,
    "gemini": true,
    "hybrid_service": true,
    "ml_model": true,
    "sentence_transformer": true,
    "feedback_collector": true,
    "retrain_queue": true
  },
  "required_services": {
    "mediapipe": true,
    "gemini": true,
    "hybrid_service": true
  },
  "optional_services": {
    "ml_model": true,
    "sentence_transformer": true,
    "feedback_collector": true,
    "retrain_queue": true
  }
}
```

**Status Values:**
- `"healthy"` - All required services are running
- `"degraded"` - Some optional services failed (server still functional)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Desktop
- AWS CLI configured
- Google Gemini API key

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/hairme-backend.git
cd hairme-backend
```

### 2. Set Up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file
nano .env
```

**Required environment variables:**
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

### 3. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Run Locally

```bash
# Start development server
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Test the API:**
```bash
# Health check
curl http://localhost:8000/api/health

# Analyze image
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@test_image.jpg"
```

---

## 💻 Local Development

### Development Setup

#### Option 1: Local Python

```bash
# Activate virtual environment
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run with hot reload
uvicorn main:app --reload --log-level debug
```

#### Option 2: Docker

```bash
# Build Docker image
docker build -t hairme-backend:dev .

# Run container
docker run -p 8000:8000 \
  -e GEMINI_API_KEY=$GEMINI_API_KEY \
  -e USE_DYNAMODB=false \
  hairme-backend:dev
```

#### Option 3: Lambda Local Testing

```bash
# Build Lambda container
docker build -f Dockerfile.lambda -t hairme-lambda:local .

# Run Lambda runtime locally
./scripts/test_lambda_local.sh --verbose
```

### Database Setup

#### DynamoDB Local Development

**Option 1: Use AWS DynamoDB (Recommended)**
```bash
# Create DynamoDB table
./scripts/create_dynamodb_table.sh

# Test connection
python scripts/test_dynamodb_connection.py
```

**Option 2: Use DynamoDB Local**
```bash
# Download and run DynamoDB Local
docker run -p 8000:8000 amazon/dynamodb-local

# Update .env
DYNAMODB_ENDPOINT=http://localhost:8000
```

**Option 3: Use MySQL (Legacy)**
```bash
# Update .env
USE_DYNAMODB=false
DATABASE_URL=mysql+pymysql://user:pass@localhost:3306/hairme

# Initialize MySQL
mysql -u root -p < db_schema_v20.sql
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_dynamodb_integration.py -v

# Run integration tests only
pytest tests/ -v -m integration
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

---

## 🚢 Deployment

### Deploy to AWS Lambda

#### Automated Deployment (Recommended)

```bash
# Deploy with single command
./scripts/deploy_lambda.sh
```

**What it does:**
1. ✅ Validates prerequisites (AWS CLI, Docker)
2. ✅ Creates ECR repository if needed
3. ✅ Backs up previous Lambda version
4. ✅ Builds Docker image (linux/amd64)
5. ✅ Pushes to ECR
6. ✅ Updates Lambda function code & configuration
7. ✅ Sets environment variables

#### Manual Deployment

```bash
# 1. Build Docker image for Lambda
docker build -f Dockerfile.lambda \
  -t hairme-lambda:latest \
  --platform linux/amd64 .

# 2. Tag and push to ECR
aws ecr get-login-password --region ap-northeast-2 | \
  docker login --username AWS --password-stdin \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com

docker tag hairme-lambda:latest \
  <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest

docker push <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest

# 3. Update Lambda function
aws lambda update-function-code \
  --function-name hairme-analyze \
  --image-uri <account-id>.dkr.ecr.ap-northeast-2.amazonaws.com/hairme-lambda:latest \
  --region ap-northeast-2

# 4. Update environment variables
aws lambda update-function-configuration \
  --function-name hairme-analyze \
  --environment "Variables={
    GEMINI_API_KEY=$GEMINI_API_KEY,
    USE_DYNAMODB=true,
    AWS_REGION=ap-northeast-2,
    DYNAMODB_TABLE_NAME=hairme-analysis,
    LOG_LEVEL=INFO
  }" \
  --region ap-northeast-2
```

#### Deployment Options

```bash
# Dry run (show what would be deployed)
./scripts/deploy_lambda.sh --dry-run

# Deploy with custom settings
./scripts/deploy_lambda.sh \
  --function-name hairme-prod \
  --memory 4096 \
  --timeout 60

# Create new function (first deployment)
./scripts/deploy_lambda.sh --create
```

### Verify Deployment

```bash
# Check Lambda function status
aws lambda get-function \
  --function-name hairme-analyze \
  --region ap-northeast-2

# Test Lambda function
aws lambda invoke \
  --function-name hairme-analyze \
  --payload '{"resource":"/api/health","path":"/api/health","httpMethod":"GET"}' \
  output.json

cat output.json | jq '.'

# Monitor logs
aws logs tail /aws/lambda/hairme-analyze --follow
```

---

## 📚 API Documentation

### POST /api/analyze

Analyze facial features and get personalized hairstyle recommendations.

**Request:**
```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@face_image.jpg"
```

**Response:**
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

Submit user feedback for a specific hairstyle recommendation.

**Request:**
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

**Response:**
```json
{
  "success": true,
  "message": "피드백이 성공적으로 저장되었습니다"
}
```

### GET /api/stats/feedback

Get aggregated feedback statistics.

**Request:**
```bash
curl http://localhost:8000/api/stats/feedback
```

**Response:**
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

**Full API documentation:** See [docs/API_CHANGES.md](docs/API_CHANGES.md)

---

## 🗄️ Database Schema

### DynamoDB Table: `hairme-analysis`

**Primary Key:**
- `analysis_id` (String, UUID) - Partition key

**Global Secondary Index:**
- `created_at-index` - Sort by creation time
  - Partition key: `entity_type` (String, always "ANALYSIS")
  - Sort key: `created_at` (String, ISO 8601)

**Attributes (36 total):**

| Attribute | Type | Description |
|-----------|------|-------------|
| `analysis_id` | String | UUID primary key |
| `created_at` | String | ISO 8601 timestamp |
| `image_hash` | String | SHA-256 hash of image |
| `face_shape` | String | Detected face shape |
| `gemini_shape` | String | Gemini's interpretation |
| `recommended_styles` | List | Array of style objects |
| `style_1_feedback` | String | "good" / "bad" / null |
| `style_1_naver_clicked` | Boolean | Search link clicked |
| `mediapipe_*` | Number | Face measurements |
| `processing_time` | Number | Analysis duration (seconds) |

**Example Item:**
```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "entity_type": "ANALYSIS",
  "created_at": "2025-01-16T12:34:56.789Z",
  "image_hash": "a1b2c3d4...",
  "face_shape": "oval",
  "gemini_shape": "타원형",
  "recommended_styles": [
    {
      "style_name": "레이어드 컷",
      "reason": "타원형 얼굴에 잘 어울립니다",
      "image_search_url": "https://..."
    }
  ],
  "style_1_feedback": "good",
  "style_1_naver_clicked": true,
  "mediapipe_face_ratio": "1.35",
  "processing_time": "2.45"
}
```

**Schema visualization:** See [docs/MIGRATION_COMPLETE.md](docs/MIGRATION_COMPLETE.md#database-schema-comparison)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Lambda Cold Start is Slow

**Symptom:** First request takes 5-10 seconds

**Solution:**
```bash
# Option 1: Use provisioned concurrency (costs money)
aws lambda put-provisioned-concurrency-config \
  --function-name hairme-analyze \
  --provisioned-concurrent-executions 1

# Option 2: Accept cold starts (most requests are warm)
# No action needed - cold starts only occur after 15 min inactivity
```

#### 2. DynamoDB Connection Error

**Symptom:** `❌ DynamoDB not initialized`

**Solution:**
```bash
# 1. Check environment variables
grep USE_DYNAMODB .env
# Expected: USE_DYNAMODB=true

# 2. Check AWS credentials
aws sts get-caller-identity

# 3. Check table exists
aws dynamodb describe-table \
  --table-name hairme-analysis \
  --region ap-northeast-2

# 4. Test connection
python scripts/test_dynamodb_connection.py
```

#### 3. Gemini API Error

**Symptom:** `❌ Gemini API call failed`

**Solution:**
```bash
# 1. Check API key is set
echo $GEMINI_API_KEY

# 2. Test API key
curl https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=$GEMINI_API_KEY \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"parts":[{"text":"test"}]}]}'

# 3. Check quota
# Visit: https://aistudio.google.com/app/apikey
```

#### 4. Docker Build Fails on Windows

**Symptom:** `platform linux/amd64 does not match`

**Solution:**
```bash
# Enable Docker BuildKit
set DOCKER_BUILDKIT=1

# Build with explicit platform
docker build -f Dockerfile.lambda \
  --platform linux/amd64 \
  -t hairme-lambda:latest .
```

#### 5. Lambda Function Not Found

**Symptom:** `ResourceNotFoundException: Function not found`

**Solution:**
```bash
# Create Lambda function first
./scripts/deploy_lambda.sh --create

# Or create manually
aws lambda create-function \
  --function-name hairme-analyze \
  --package-type Image \
  --code ImageUri=<ecr-image-uri> \
  --role arn:aws:iam::<account-id>:role/hairme-lambda-role \
  --region ap-northeast-2
```

#### 6. MediaPipe Import Error

**Symptom:** `ImportError: libGL.so.1: cannot open shared object file`

**Solution:**
Already fixed in `Dockerfile.lambda`:
```dockerfile
RUN yum install -y mesa-libGL
```

If still occurring, rebuild Docker image.

#### 7. Float to Decimal Error (DynamoDB)

**Symptom:** `TypeError: Float types are not supported`

**Solution:**
Already fixed in `database/dynamodb_connection.py`:
```python
def _convert_floats_to_decimal(obj):
    # Automatically converts float to Decimal
```

No action needed - handled by conversion function.

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python -m uvicorn main:app --log-level debug

# Check CloudWatch Logs (Lambda)
aws logs tail /aws/lambda/hairme-analyze --follow --filter-pattern "ERROR"
```

### Performance Optimization

```bash
# Monitor DynamoDB capacity
aws cloudwatch get-metric-statistics \
  --namespace AWS/DynamoDB \
  --metric-name ConsumedReadCapacityUnits \
  --dimensions Name=TableName,Value=hairme-analysis \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Sum

# Monitor Lambda performance
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=hairme-analyze \
  --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 \
  --statistics Average,Maximum
```

---

## 📖 Documentation

### Complete Documentation

- [MIGRATION_COMPLETE.md](docs/MIGRATION_COMPLETE.md) - Architecture, cost analysis, performance comparison
- [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) - Step-by-step migration from RDS to DynamoDB
- [ROLLBACK.md](ROLLBACK.md) - Emergency recovery procedures
- [API_CHANGES.md](docs/API_CHANGES.md) - Android app integration guide
- [.env.example](.env.example) - Environment variables template

### Scripts

| Script | Description |
|--------|-------------|
| `scripts/create_dynamodb_table.sh` | Create DynamoDB table |
| `scripts/test_dynamodb_connection.py` | Test database connection |
| `scripts/deploy_lambda.sh` | Deploy to AWS Lambda |
| `scripts/test_lambda_local.sh` | Test Lambda locally |
| `scripts/cleanup_infrastructure.sh` | Remove old AWS resources |
| `scripts/verify_cleanup.sh` | Verify cost savings |

---

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes
4. Run tests (`pytest tests/ -v`)
5. Format code (`black .`)
6. Commit changes (`git commit -m 'Add amazing feature'`)
7. Push to branch (`git push origin feature/amazing-feature`)
8. Open Pull Request

### Code Standards

- Python 3.11+
- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions
- Maintain test coverage > 80%
- Update documentation

---

## 📊 Project Status

**Current Version:** 20.2.0 (DynamoDB + Lambda)
**Architecture:** Serverless (AWS Lambda + DynamoDB)
**Status:** ✅ Production Ready
**Monthly Cost:** ~$0.00 (within Free Tier)

### Recent Updates

- ✅ Migrated from RDS MySQL to DynamoDB (100% data migrated)
- ✅ Deployed serverless architecture (Lambda + DynamoDB)
- ✅ Reduced costs by 95% ($61/month → $0/month)
- ✅ Improved query performance by 20-75%
- ✅ Implemented comprehensive testing suite
- ✅ Added complete documentation

### Roadmap

- [ ] API Gateway integration (rate limiting, caching)
- [ ] CloudWatch alarms for monitoring
- [ ] CI/CD with GitHub Actions
- [ ] Multi-region deployment
- [ ] GraphQL API layer

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

**Developer:** HairMe Development Team
**Contact:** [your-email@example.com](mailto:your-email@example.com)
**GitHub:** [your-repo/hairme-backend](https://github.com/your-repo/hairme-backend)

---

## 🙏 Acknowledgments

- **Google Gemini** - AI-powered recommendations
- **MediaPipe** - Face mesh analysis
- **FastAPI** - Modern Python web framework
- **AWS** - Serverless infrastructure
- **Claude Code** - Development assistance

---

**Made with ❤️ by HairMe Team**
