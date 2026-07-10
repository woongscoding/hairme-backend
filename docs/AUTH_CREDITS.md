# 회원 인증 + 크레딧 + 사진 저장 시스템

플랫폼 전환(분석 → 예약 → 커머스)의 1단계 기반. 카카오 로그인, 크레딧 과금, S3 사진 저장을 제공한다.

## 아키텍처

```
Android App
  │ ① 카카오 SDK 로그인 → 카카오 액세스 토큰
  ▼
POST /api/auth/kakao ──② 토큰 검증──▶ kapi.kakao.com/v2/user/me
  │ ③ 회원 조회/생성 (hairme-users, 신규 가입 시 보너스 5크레딧)
  │ ④ 자체 JWT 발급 (access 1h + refresh 30d)
  ▼
POST /api/v2/synthesize (Authorization: Bearer <JWT>)
  │ ⑤ 캐시 확인 (S3 cache/, 히트 시 과금 없이 반환)
  │ ⑥ 크레딧 1 차감 (DynamoDB 조건부 원자 업데이트, 부족 시 402)
  │ ⑦ Gemini 합성 (실패 시 자동 환불)
  │ ⑧ S3 저장: cache/ + results/{user_id}/ + (동의 시) originals/{user_id}/
  ▼
응답: image_base64 + result_url + quota
```

## 데이터 모델

### DynamoDB `hairme-users`
| 속성 | 타입 | 설명 |
|------|------|------|
| `user_id` (PK) | S | UUID |
| `kakao_id` (GSI: kakao_id-index) | S | 카카오 회원번호 |
| `credits` | N | 크레딧 잔액 (조건부 업데이트로만 변경) |
| `training_consent` | BOOL | 원본 사진 AI 학습 활용 동의 (기본 false, 별도 opt-in) |
| `nickname`, `email`, `status`, `created_at`, `last_login_at` | | |

### DynamoDB `hairme-credit-ledger` (감사/CS용 원장)
| 속성 | 타입 | 설명 |
|------|------|------|
| `user_id` (PK) | S | |
| `sk` (SK) | S | `{ISO8601}#{txid}` — 시간순 정렬 |
| `amount` | N | 증감량 (+지급/-차감) |
| `reason` | S | signup_bonus / synthesis / refund / purchase / admin_grant |
| `balance_after` | N | 처리 후 잔액 |

### S3 버킷 (PHOTO_S3_BUCKET)
| Prefix | 용도 | 수명주기 |
|--------|------|---------|
| `cache/{sha256}` | 합성 결과 캐시 — 같은 사진+스타일 재요청 시 Gemini 재호출 방지 (비용 절감 핵심) | 90일 |
| `results/{user_id}/` | 회원별 합성 결과 (presigned URL로 재열람) | 180일 |
| `originals/{user_id}/` | AI 학습용 원본 (training_consent=true 회원만) | 무기한 |

퍼블릭 액세스 전면 차단, presigned URL로만 접근.

## API

| 메서드 | 엔드포인트 | 인증 | 설명 |
|--------|-----------|------|------|
| POST | `/api/auth/kakao` | - | 카카오 로그인/가입 (JWT 발급) |
| POST | `/api/auth/refresh` | refresh token | 액세스 토큰 재발급 |
| GET | `/api/auth/me` | JWT | 프로필 + 크레딧 조회 |
| PATCH | `/api/auth/me/consent` | JWT | 학습 활용 동의 변경 |
| GET | `/api/credits` | JWT | 잔액 + 최근 내역 |
| POST | `/api/credits/purchase` | JWT | 인앱결제 (Google Play 영수증 검증, purchase_token 멱등성) |
| GET | `/api/credits/reward-callback` | ECDSA 서명 | AdMob 리워드 광고 SSV 콜백 (+1 크레딧, 일일 상한) |
| GET | `/api/me/results` | JWT | 합성 결과 히스토리 (최신순, presigned URL, 페이지네이션) |
| POST | `/api/admin/credits/grant` | Admin Key | 수동 크레딧 지급 |

`/api/v2/synthesize`, `/api/v2/synthesize-with-reference`: JWT 있으면 크레딧 차감,
없으면 기존 device_id 일일 제한 (구버전 앱 호환, 단계적 폐기 예정).

## 배포 절차

```bash
# 1. AWS 리소스 생성 (DynamoDB 테이블 2개 + S3 버킷)
python scripts/create_auth_tables.py --create-bucket hairme-photos

# 2. JWT 시크릿 생성 및 등록
python -c "import secrets; print(secrets.token_urlsafe(64))"
aws secretsmanager create-secret --name hairme-jwt-secret --secret-string '<시크릿>'

# 3. 환경 변수
#    PHOTO_S3_BUCKET=hairme-photos
#    (로컬은 .env에 JWT_SECRET_KEY 직접 설정)

# 4. Lambda IAM 역할에 권한 추가
#    dynamodb: hairme-users, hairme-credit-ledger (GetItem/PutItem/UpdateItem/Query)
#    s3: hairme-photos (GetObject/PutObject)
```

## 설계 결정 (면접 어필 포인트)

1. **크레딧 차감은 조건부 원자 업데이트** — `ConditionExpression: credits >= :amt`로
   동시 요청에서도 음수 잔액 불가. 별도 락/트랜잭션 불필요.
2. **캐시 확인을 과금보다 먼저** — 캐시 히트는 원가 0이므로 과금하지 않음.
3. **합성 실패 시 자동 환불** — 크레딧 모드만. 원장에 refund로 기록되어 추적 가능.
4. **만료 토큰은 익명 폴백 없이 401** — 만료 토큰으로 무료(device_id) 흐름을 타는
   우회를 차단.
5. **원본 사진은 opt-in 저장** — 개인정보보호법상 AI 학습 활용은 서비스 제공과 별도
   선택 동의가 필요. 기본은 저장하지 않고, 동의 회원만 originals/에 보관.
6. **원장(ledger)은 best-effort** — 원장 기록 실패가 결제 흐름을 막지 않음.
   잔액(단일 진실)은 users 테이블, 원장은 감사용.
