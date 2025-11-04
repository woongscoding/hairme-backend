# HairMe MLOps 파이프라인

실제 사용자 데이터로 지속적으로 학습하는 자동화된 MLOps 시스템입니다.

## 📋 목차

- [개요](#개요)
- [아키텍처](#아키텍처)
- [설치](#설치)
- [사용법](#사용법)
- [스크립트 상세](#스크립트-상세)
- [자동화 설정](#자동화-설정)
- [문제 해결](#문제-해결)

---

## 개요

HairMe 앱에서 사용자가 제공한 피드백(좋아요/싫어요)을 수집하여, 합성 데이터로 학습된 모델을 실제 데이터로 지속적으로 개선하는 시스템입니다.

### 주요 기능

- ✅ **자동 데이터 추출**: DB에서 피드백 데이터 자동 추출
- ✅ **데이터 병합**: 합성 데이터 + 실제 데이터 결합
- ✅ **자동 재학습**: 새 데이터로 모델 재학습
- ✅ **성능 평가**: 기존 모델과 성능 비교
- ✅ **안전한 배포**: 성능 개선 시에만 자동 배포
- ✅ **버전 관리**: 모델 버전 및 백업 관리

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    MLOps 파이프라인                         │
└─────────────────────────────────────────────────────────────┘

1. 데이터 추출 (export_real_data.py)
   ↓
   - MySQL RDS에서 피드백 데이터 조회
   - 학습 데이터 형식으로 변환
   - CSV로 저장

2. 데이터 준비 (prepare_training_data.py)
   ↓
   - 합성 데이터 + 실제 데이터 병합
   - 실제 데이터 가중치 적용 (2배 증폭)
   - Train/Val/Test 분할 (70/15/15)

3. 모델 재학습 (retrain_model.py)
   ↓
   - PyTorch 모델 학습
   - Early stopping 적용
   - 클래스 불균형 해결 (Class Weights)
   - 체크포인트 및 메트릭 저장

4. 모델 배포 (deploy_model.py)
   ↓
   - 새 모델 vs 기존 모델 성능 비교
   - F1-score 기준 평가
   - 성능 개선 시 프로덕션 배포
   - 기존 모델 백업

5. 결과
   ↓
   - models/final_model.pth ← 새 모델
   - models/encoders.pkl ← 새 인코더
   - 배포 히스토리 기록
```

---

## 설치

### 1. 필요한 패키지 설치

이미 `requirements.txt`에 포함되어 있습니다:

```bash
pip install torch pandas numpy scikit-learn pymysql matplotlib seaborn pyyaml
```

### 2. 환경변수 설정

MLOps 파이프라인이 DB에 접근하려면 환경변수가 필요합니다:

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL="asyncmy://admin@hairme-data.xxx.rds.amazonaws.com:3306/hairme"
$env:DB_PASSWORD="your_password"
```

**Linux/Mac:**
```bash
export DATABASE_URL="asyncmy://admin@hairme-data.xxx.rds.amazonaws.com:3306/hairme"
export DB_PASSWORD="your_password"
```

또는 `.env` 파일 생성:
```
DATABASE_URL=asyncmy://admin@hairme-data.xxx.rds.amazonaws.com:3306/hairme
DB_PASSWORD=your_password
```

---

## 사용법

### 🚀 전체 파이프라인 실행 (권장)

가장 간단한 방법입니다. 모든 단계를 자동으로 실행합니다:

```bash
python scripts/mlops/mlops_pipeline.py
```

**옵션:**
```bash
# 최소 피드백 개수 설정 (기본: 50)
python scripts/mlops/mlops_pipeline.py --min-feedback 100

# 실제 데이터 가중치 설정 (기본: 2.0)
python scripts/mlops/mlops_pipeline.py --real-weight 3.0

# 최소 성능 개선폭 설정 (기본: 0.0)
python scripts/mlops/mlops_pipeline.py --min-improvement 0.01

# 자동 배포 비활성화 (평가만)
python scripts/mlops/mlops_pipeline.py --no-auto-deploy

# 데이터 개수 확인 스킵 (강제 실행)
python scripts/mlops/mlops_pipeline.py --skip-data-check
```

### 📝 개별 스크립트 실행

각 단계를 개별적으로 실행할 수도 있습니다:

#### 1️⃣ 데이터 추출
```bash
python scripts/mlops/export_real_data.py
```
- 출력: `data_source/real_user_data_YYYYMMDD_HHMMSS.csv`
- 최신 버전: `data_source/real_user_data_latest.csv`

#### 2️⃣ 데이터 준비
```bash
python scripts/mlops/prepare_training_data.py --real-weight 2.0
```
- 출력:
  - `data_source/train_data.csv`
  - `data_source/val_data.csv`
  - `data_source/test_data.csv`

#### 3️⃣ 모델 재학습
```bash
python scripts/mlops/retrain_model.py --batch-size 64 --epochs 50
```
- 출력:
  - `models/checkpoints/model_YYYYMMDD_HHMMSS.pth`
  - `models/checkpoints/encoders_YYYYMMDD_HHMMSS.pkl`
  - `models/checkpoints/model_latest.pth`

#### 4️⃣ 모델 배포
```bash
python scripts/mlops/deploy_model.py --min-improvement 0.0
```
- 성능 개선 시:
  - `models/final_model.pth` ← 업데이트
  - `models/encoders.pkl` ← 업데이트
  - `models/backups/` ← 기존 모델 백업

---

## 스크립트 상세

### 1. `export_real_data.py`

**기능:**
- MySQL RDS의 `analysis_history` 테이블에서 피드백 데이터 추출
- JSON 필드 파싱 및 변환
- 합성 데이터와 동일한 형식으로 저장

**주요 변환:**
- `personal_color` → `skin_tone` (봄웜/가을웜 → 웜톤)
- `recommended_styles` (JSON) → 각 스타일별 행 생성
- 피드백이 있는 스타일만 추출

**출력 형식:**
```csv
face_shape,skin_tone,hairstyle,score,feedback,naver_clicked,reason
계란형,쿨톤,시스루뱅 긴머리,0.92,like,True,우아한 스타일
```

### 2. `prepare_training_data.py`

**기능:**
- 합성 데이터와 실제 데이터 병합
- 실제 데이터 가중치 적용 (중요도 증폭)
- 학습/검증/테스트 분할

**가중치 적용 예시:**
- 합성 데이터: 10,000건
- 실제 데이터: 200건
- 가중치 2.0 적용 → 실제 데이터 400건으로 증폭
- 최종: 10,400건

**클래스 불균형 체크:**
- Like/Dislike 비율 분석
- 경고 메시지 출력

### 3. `retrain_model.py`

**기능:**
- PyTorch 모델 학습
- Multi-task Learning (Score 예측 + Feedback 분류)
- Early Stopping
- 학습 곡선 시각화

**모델 구조:**
```
Embedding Layer (Face + Skin + Style)
    ↓
Shared Layers (FC + ReLU + Dropout)
    ↓
    ├─→ Score Head (Regression)
    └─→ Feedback Head (Classification)
```

**손실 함수:**
```python
Loss = MSE(score) + 2.0 × CrossEntropy(feedback, class_weights)
```

### 4. `deploy_model.py`

**기능:**
- 테스트 데이터로 성능 평가
- 기존 모델과 비교
- 성능 개선 시 배포

**평가 메트릭:**
- Accuracy
- Precision
- Recall
- **F1-Score** (배포 결정 기준)

**안전 장치:**
- 기존 모델 자동 백업
- 배포 히스토리 기록 (JSON)
- 최소 개선폭 설정 가능

### 5. `mlops_pipeline.py`

**기능:**
- 전체 파이프라인 자동 실행
- 피드백 데이터 개수 확인
- 각 단계 성공/실패 체크
- 로그 기록

**실행 조건:**
```python
if feedback_count >= min_feedback_count:
    run_pipeline()
else:
    skip()
```

---

## 자동화 설정

### 방법 1: Cron (Linux/Mac)

매주 일요일 새벽 2시에 실행:

```bash
crontab -e
```

다음 라인 추가:
```
0 2 * * 0 cd /path/to/Hairstyle_server && /path/to/python scripts/mlops/mlops_pipeline.py >> logs/mlops/cron.log 2>&1
```

### 방법 2: Windows Task Scheduler

1. "작업 스케줄러" 열기
2. "기본 작업 만들기" 클릭
3. 트리거: 매주 일요일 새벽 2시
4. 작업: Python 스크립트 실행
   - 프로그램: `python.exe` 경로
   - 인수: `scripts/mlops/mlops_pipeline.py`
   - 시작 위치: 프로젝트 루트

### 방법 3: AWS Lambda (클라우드)

Lambda 함수를 생성하여 ECS Task를 트리거:

```python
import boto3

def lambda_handler(event, context):
    ecs = boto3.client('ecs')

    response = ecs.run_task(
        cluster='hairme-cluster',
        taskDefinition='mlops-pipeline',
        launchType='FARGATE',
        # ... 네트워크 설정
    )

    return response
```

**EventBridge 규칙:**
- Schedule: `cron(0 2 ? * SUN *)`  # 매주 일요일 새벽 2시 UTC

---

## 설정 파일

`scripts/mlops/config.yaml`에서 설정 변경 가능:

```yaml
data:
  min_feedback_count: 50  # 최소 피드백 개수
  real_data_weight: 2.0   # 실제 데이터 가중치

training:
  batch_size: 64
  max_epochs: 50
  learning_rate: 0.001

deployment:
  auto_deploy: true
  min_improvement: 0.0    # 최소 F1 개선폭
```

---

## 디렉토리 구조

```
Hairstyle_server/
├── scripts/
│   └── mlops/
│       ├── export_real_data.py      # 데이터 추출
│       ├── prepare_training_data.py # 데이터 준비
│       ├── retrain_model.py         # 재학습
│       ├── deploy_model.py          # 배포
│       ├── mlops_pipeline.py        # 전체 파이프라인
│       ├── config.yaml              # 설정 파일
│       └── README.md                # 이 문서
│
├── models/
│   ├── final_model.pth              # 프로덕션 모델
│   ├── encoders.pkl                 # 프로덕션 인코더
│   ├── checkpoints/                 # 학습된 모델들
│   │   ├── model_YYYYMMDD.pth
│   │   └── encoders_YYYYMMDD.pkl
│   └── backups/                     # 백업 모델들
│
├── data_source/
│   ├── synthetic_hairstyle_data.csv # 합성 데이터
│   ├── real_user_data_latest.csv    # 실제 데이터
│   ├── train_data.csv               # 학습 데이터
│   ├── val_data.csv                 # 검증 데이터
│   └── test_data.csv                # 테스트 데이터
│
└── logs/
    └── mlops/
        ├── pipeline_YYYYMMDD.log    # 파이프라인 로그
        └── training/                # 학습 로그
```

---

## 문제 해결

### 1. "환경변수가 설정되지 않았습니다"

**원인:** `DATABASE_URL` 또는 `DB_PASSWORD` 환경변수 없음

**해결:**
```bash
export DATABASE_URL="asyncmy://admin@..."
export DB_PASSWORD="..."
```

### 2. "피드백 데이터가 없습니다"

**원인:** DB에 피드백 데이터가 50개 미만

**해결:**
- `--skip-data-check` 옵션으로 강제 실행
- 또는 `--min-feedback 10`으로 최소 개수 조정

### 3. "모델 성능이 개선되지 않아 배포 거부"

**원인:** 새 모델의 F1-score가 기존 모델보다 낮음

**해결:**
- 정상 동작입니다 (안전 장치)
- 더 많은 데이터를 수집한 후 재시도
- 또는 `--min-improvement -0.01`로 임계값 낮추기

### 4. "인코더 변환 실패"

**원인:** 새로운 카테고리 값이 추가됨 (예: 새로운 헤어스타일)

**해결:**
- `data_source/synthetic_hairstyle_data.csv`에 해당 카테고리 추가
- 또는 데이터 전처리 코드에서 Unknown 카테고리 처리

### 5. 모델 배포 후 서버에서 로드 실패

**원인:** 인코더 클래스 개수가 달라짐

**해결:**
```bash
# 서버 재시작
docker restart hairme-server

# 또는 ECS에서
aws ecs update-service --cluster hairme --service hairme-api --force-new-deployment
```

---

## 모니터링

### 배포 히스토리 확인

```bash
cat models/deployment_history.json
```

예시:
```json
[
  {
    "timestamp": "2024-01-15T02:00:00",
    "deployed": true,
    "new_model_metrics": {
      "f1": 0.89,
      "accuracy": 0.87
    },
    "current_model_metrics": {
      "f1": 0.85,
      "accuracy": 0.83
    }
  }
]
```

### 학습 로그 확인

```bash
tail -f logs/mlops/pipeline_YYYYMMDD.log
```

### 모델 성능 추적

```python
import json
with open('models/deployment_history.json') as f:
    history = json.load(f)

for record in history:
    print(f"{record['timestamp']}: F1 = {record['new_model_metrics']['f1']:.4f}")
```

---

## 다음 단계

- [ ] Slack/Email 알림 추가
- [ ] A/B 테스트 프레임워크
- [ ] 데이터 드리프트 감지
- [ ] 모델 설명 가능성 (SHAP)
- [ ] Hyperparameter Tuning 자동화

---

## 라이센스

MIT License

---

## 문의

문제가 있으면 이슈를 남겨주세요!
