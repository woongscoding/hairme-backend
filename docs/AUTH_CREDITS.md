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
| `status` | S | `active` / `suspended` (속성이 없으면 active 로 취급) |
| `token_version` | N | 리프레시 토큰 세대 번호 (기본 1, 속성이 없으면 1로 취급) |
| `nickname`, `email`, `created_at`, `last_login_at` | | |

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
| POST | `/api/auth/refresh` | refresh token | 액세스 + 리프레시 토큰 재발급 (계정/세대 검증) |
| POST | `/api/auth/logout-all` | JWT | 모든 기기 로그아웃 (token_version 증가) |
| GET | `/api/auth/me` | JWT | 프로필 + 크레딧 조회 |
| PATCH | `/api/auth/me/consent` | JWT | 학습 활용 동의 변경 |
| GET | `/api/credits` | JWT | 잔액 + 최근 내역 |
| POST | `/api/credits/purchase` | JWT | 인앱결제 (Google Play 영수증 검증, purchase_token 멱등성) |
| GET | `/api/credits/reward-callback` | ECDSA 서명 | AdMob 리워드 광고 SSV 콜백 (+1 크레딧, 일일 상한) |
| GET | `/api/me/results` | JWT | 합성 결과 히스토리 (최신순, presigned URL, 페이지네이션) |
| POST | `/api/admin/credits/grant` | Admin Key | 수동 크레딧 지급 |
| POST | `/api/admin/users/{user_id}/suspend` | Admin Key | 계정 정지 (status=suspended + token_version 증가) |
| POST | `/api/admin/users/{user_id}/reactivate` | Admin Key | 계정 정지 해제 (status=active) |

`/api/v2/synthesize`, `/api/v2/synthesize-with-reference`: JWT 있으면 크레딧 차감,
없으면 기존 device_id 일일 제한 (구버전 앱 호환, 단계적 폐기 예정 - 아래 킬 스위치 참고).

## 세션 무효화 (token_version)

JWT 는 상태가 없어 "발급된 토큰 회수"가 기본적으로 불가능하다. 그래서 사용자
아이템에 세대 번호(`token_version`)를 두고, 액세스/리프레시 토큰 **양쪽에 `tv`
클레임**을 심는다.

```
POST /api/auth/kakao      → tv = 사용자의 token_version 으로 access/refresh 발급
POST /api/auth/refresh    → refresh 의 tv == 사용자 token_version 인지 대조 (강한 일관성 조회)
POST /api/auth/logout-all → token_version += 1  →  이전 refresh 토큰 전부 401
```

| 지점 | DB 조회 | 근거 |
|------|---------|------|
| 액세스 토큰 검증 (`get_current_user_id`) | **없음 (stateless)** | 모든 요청마다 users 테이블을 읽으면 지연/비용이 커진다. 액세스 토큰이 짧기(기본 60분) 때문에 허용되는 트레이드오프 |
| 리프레시 (`POST /api/auth/refresh`) | **강한 일관성 get_item** | 방금 바뀐 token_version/status 를 반드시 봐야 하므로 `ConsistentRead=True` |

**호환성**: `tv` 클레임이 없는 과거 토큰은 `tv=1`, `token_version` 속성이 없는
기존 회원도 `1` 로 간주한다. 따라서 이 변경 이후에도 기존 세션은 그대로 유지되고,
첫 무효화(bump)에서 2가 되며 그 순간 과거 토큰이 모두 죽는다.

### 계정 정지 (suspend)

| 경로 | 정지 계정의 결과 |
|------|-----------------|
| `POST /api/auth/kakao` | **403** (토큰 미발급, 한국어 안내 메시지) |
| `POST /api/auth/refresh` | **401** |
| 액세스 토큰이 필요한 일반 엔드포인트 | **정지 전에 발급된 액세스 토큰은 만료까지 그대로 동작** |

즉 정지는 **최대 `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`(기본 60분)의 지연**을 가진다.
액세스 토큰 검증을 stateless 로 유지한 대가이며, 즉시 차단이 필요한 사고(계정
탈취·결제 어뷰징)에서는 정지와 함께 크레딧을 0으로 만들거나 해당 기능을 막는
쪽이 확실하다. `suspend` 는 status 변경과 token_version 증가를 함께 수행하므로
재로그인 경로는 즉시 막힌다.

`reactivate` 는 status 만 `active` 로 되돌리고 token_version 은 되돌리지 않는다
(정지 중 무효화된 세션은 그대로 폐기 - 사용자는 다시 로그인해야 한다).

## 레거시 device_id 흐름 킬 스위치

비로그인 device_id 무료 흐름은 단계적 폐기 대상이다. `LEGACY_DEVICE_FLOW_ENABLED`
(기본 `True`) 하나로 전체를 끈다.

| 설정 | 합성 계열 (비로그인) | `/api/usage`, `/api/usage/consume` |
|------|---------------------|-----------------------------------|
| `True` (기본) | 기존 device + IP 일일 제한 동작 | 정상 |
| `False` | **401** "로그인이 필요합니다…" (카운터를 건드리기 전에 차단) | **410 Gone** |

플래그를 내리기 전에 잔존 사용량을 측정할 수 있도록, 익명 과금 1건마다
`core.quota` 가 구조화 로그 1줄을 남긴다. **원본 device_id / IP 는 절대 남기지
않는다** (device_id = 앞 4자 + sha256 앞 8자, IPv4 = 마지막 옥텟 0, IPv6 = /48).

```json
{"event_type": "legacy_device_flow", "endpoint": "synthesize",
 "device_id": "a1b2~9f86d081", "ip": "203.0.113.0"}
```

CloudWatch Logs Insights 예시:

```
fields @timestamp, endpoint, device_id
| filter event_type = "legacy_device_flow"
| stats count() as calls, count_distinct(device_id) as devices by endpoint, bin(1d)
```

일일 호출/기기 수가 충분히 0에 수렴하면 `LEGACY_DEVICE_FLOW_ENABLED=false` 로
배포한다 (환경변수만 바꾸면 되고 코드 변경/롤백이 필요 없다).

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
7. **세션 무효화는 token_version 세대 번호로** — 토큰 블랙리스트(모든 요청마다
   조회)를 만들지 않고, 검증 비용을 리프레시 시점 1회로 몰았다. 짧은 액세스 토큰
   수명이 "최대 60분 지연"이라는 비용의 상한을 정한다.
8. **레거시 흐름은 삭제 전에 계측** — 마스킹된 구조화 로그로 실제 잔존 사용량을
   측정한 뒤, 코드 삭제가 아니라 환경변수 플래그로 끈다 (즉시 롤백 가능).
