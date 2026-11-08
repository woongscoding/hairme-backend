# 수익화 v2.1 — 앱이 기대하는 서버 계약

앱(Phase 1·2)은 아래 필드가 **없어도** 지금 서버에서 기존 동작으로 돌아가고, 서버가 필드를 내려주는 순간 새 동작으로 바뀐다. 서버 작업 때 이 이름 그대로 맞추면 앱 재배포 없이 켜진다.

## 이미 있는 것 (앱이 그대로 사용)

- `POST /api/credits/purchase` `{product_id, purchase_token}` → 200 `{granted, balance}` / 409 같은 토큰 중복(이미 지급) / 422 `one_per_account_used`(첫 구매 상품 재구매, 서버 미승인) / 400 무효. 서버는 acknowledge만 하므로 **앱이 200과 `error` 없는 409 뒤에만 consume** 한다.
- `GET /api/credits` → `balance`, `history[].reason` (`"purchase"`가 있으면 유료 사용자로 판정)
- 합성 응답 `quota` (`mode: credits|device`) → 잔액·남은 횟수 갱신에 사용 (`/api/usage/consume` 호출은 제거)
- 429 본문 `limit_type` → `"device"`일 때만 결제 화면, `"ip"`·분당 제한은 일반 에러
- 기기 한도 429의 `limit_detail` → `"lifetime"`(실험군 평생 무료 체험 소진)이면 로그인 유도 문구로 결제/로그인 화면, `"daily"`는 기존 문구
- 첫 구매 상품 재구매 거절 `error: "one_per_account_used"` → **consume 하지 않고**(서버 미승인 → Google 자동 환불) "결제 금액은 자동 환불돼요" 안내. 서버는 **422**로 보낸다. 앱은 상태 코드가 아니라 `error`로 판정하므로 409로 와도 consume 하지 않는다. `error`가 없는 409만 "이미 지급됨 → consume"
- `GET /api/credits/products` → `starter_5`의 `available`로 첫 구매 상품 노출 (회원만, 실패하면 결제 이력으로 판단)

## 서버에 추가해야 하는 것

| 위치 | 필드 | 값 | 앱 동작 |
| --- | --- | --- | --- |
| `GET /api/usage` | `free_policy` | `"daily"` \| `"lifetime"` | 남은 횟수·결제 화면 문구 ("오늘 무료" vs "무료 체험") |
| `GET /api/usage` | `experiment_arm` | `"control"` \| `"treatment"` | treatment: 보상형 광고가 크레딧 대신 퍼스널컬러 상세·전체 랭킹을 연다. 홈/결제 화면의 "광고 보고 크레딧 받기" 숨김. Firebase 사용자 속성 `free_policy_arm` |
| `GET /api/usage` | `signup_bonus` | int (대조군 5, 실험군 3) | "로그인하고 N회 더 받기" |
| 합성 응답 (스타일·염색) | `is_preview` | bool | true면 "무료 미리보기 · 512px · 워터마크" + "이 스타일 고화질로 보기" → 결제 화면 |
| `POST /api/v2/analyze-hybrid` `data` | `full_ranking` | `[{style_name, score}]` 점수순 | 잠금 해제 시 "전체 스타일 랭킹" 표시 |
| `GET /api/credits` | `has_purchased` | bool | history 20건 밖의 결제까지 반영 (없으면 history로 판정) |
| `CREDIT_PRODUCTS` | `starter_5` | 5 | 첫 구매 상품. **계정당 1회 제한은 서버가 강제해야 한다** (앱은 결제 이력이 있으면 숨기기만 함) |

주의할 점:
- 실험군 배정(신규 기기 해시 50:50), 512px·워터마크 렌더링, 캐시 키에 화질 포함(무료 미리보기 캐시가 유료 재합성에 적중하면 안 됨)은 모두 서버 쪽 작업.
- 실험군 기능 해제 광고는 SSV `user_id` 없이 띄운다. 서버의 `reward-callback`은 이를 `verification_ping`으로 보고 지급하지 않아야 한다 (현재 동작 그대로).

## 웹 (공유 링크)

- `https://hairme.app/.well-known/assetlinks.json` — `com.hairme.app` + Play 앱 서명 키 SHA-256
- `https://hairme.app/s?m=<section>` — 앱 없으면 Play(`referrer=utm_source=result_share&utm_medium=<section>&utm_campaign=result_share_v1`)로 리다이렉트
- 준비되면 앱의 `ShareHelper.SHORT_LINK_BASE`를 `"https://hairme.app/s"`로 바꾼다. 그 전까지 공유 텍스트는 Play 링크를 직접 쓴다.

## 앱이 남기는 Firebase 이벤트 (Phase 3 판정용)

`paywall_view`(trigger, limit_type, logged_in, credits30_variant, experiment_arm) → `purchase_start` → `purchase_complete`(sku, credits, value, currency, placement; 서버 검증 후) / `purchase_failed`,
`login`(new_user), `hd_upgrade_click`, `feature_unlock`, `salon_intent_declared`, `install_referrer`, `app_link_open`, `push_sent` / `push_open`

## 암묵 피드백 (합성·저장·공유 → 추천 모델 학습 샘플)

명시 피드백(좋아요/싫어요)만으로는 재학습 데이터가 모이지 않아, 합성과 합성 결과의 저장/공유를 학습 신호로 쓴다. 아래 필드는 전부 **선택**이며, 없으면 서버는 지금과 똑같이 동작한다(구버전 앱 호환).

### (a) 합성 요청에 추천 카드 정보 첨부

`POST /api/v2/synthesize` (multipart form, 앱이 쓰는 실제 경로). 레퍼런스 합성(`/api/v2/synthesize-with-reference`)은 호출이 없어 2026-11 에 삭제됐다.

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `analysis_id` | str | 추천 카드가 나온 분석 ID (`/api/v2/analyze-hybrid` 응답) |
| `rec_rank` | int | 화면에 표시된 카드 순위 1~5 |
| `hairstyle_id` | int | 카드의 `hairstyle_id` (트렌드 카드는 null 이므로 보내지 않음) |

- 형식이 틀린 값(숫자가 아님, 범위 밖 등)은 **무시**한다. 400/422 를 내지 않는다.
- 응답 본문은 변경 없음. 합성이 성공하면(캐시 히트 포함) 응답 뒤에서 `synthesize` 신호를 기록하며, 기록 실패는 합성 결과와 무관하다.
- `synthesis_success` 구조화 로그에 `analysis_id`, `rec_rank` 가 추가된다(없으면 null).

### (b) 저장/공유 이벤트

`POST /api/events/style-action` (JSON, 로그인 선택, 분당 30회)

```json
{"analysis_id": "uuid", "style_name": "레이어드 컷", "hairstyle_id": 42,
 "action": "save", "result_id": "합성 응답의 result_id", "device_id": "..."}
```

- `action`: `"save"` | `"share"` (앱은 공유 시트를 연 시점에 share 를 보낸다 — 취소 포함)
- `hairstyle_id`, `device_id` 는 null 가능. `result_id` 는 스키마상 null 가능하지만 **없으면 학습 샘플로 기록되지 않는다** — 합성 응답의 `result_id` 를 그대로 보낼 것
- 응답: 기록 성공 여부와 무관하게 **200 `{"success": true}`**. 스키마 위반(필수 필드 누락, 알 수 없는 action 등)만 422. 토큰이 만료·손상돼도 401 을 내지 않는다.
- 헤어컬러 합성 결과에는 호출하지 않는다.

### 서버 쪽 처리 (참고)

- 학습 샘플: `s3://hairme-mlops/feedback/pending/implicit/{analysis_id}_{hairstyle_id}.npz` — 명시 피드백 npz 와 같은 레이아웃 + `sample_weight`, metadata 에 `source:"implicit"`, `signal_type`, `rec_rank`, `ml_rank`, `propensity`
- 라벨/가중치: share 85·1.0, save 80·1.0, synthesize 55·0.7, shown_not_chosen 40·0.3. `sample_weight = base × clip(1/propensity, 1, 5)` (propensity 없으면 1.0)
- save/share 는 서버가 그 분석×스타일을 실제로 합성했고(쌍 표식 `feedback/implicit_markers/pairs/{analysis_id}_{hairstyle_id}`), 요청의 `result_id` 가 그 합성의 result_id(최근 8개) 중 하나일 때만 기록한다. 아니면 생략하고 로그 `implicit_signal` 에 reason(`no_synthesis_marker` / `result_id_missing` / `result_id_mismatch`)
- 같은 분석×스타일은 더 강한 신호일 때만 덮어쓴다 (share > save > synthesize > shown_not_chosen)
- 한 분석의 첫 synthesize 때, 나머지 ML 카드에 `shown_not_chosen` 샘플을 만든다(이미 있으면 그대로). 트렌드 카드는 샘플을 만들지 않는다.
- 카드 매칭: `hairstyle_id` → 없거나 안 맞으면 `style_name`(띄어쓰기 무시) → 그래도 없으면 기록 생략
- 분석이 없거나, 얼굴 측정값이 없거나, `MLOPS_ENABLED` 가 꺼져 있으면 조용히 생략(로그 `implicit_signal` 만 남김)
- user_id / device_id 원문은 샘플에 저장하지 않는다
