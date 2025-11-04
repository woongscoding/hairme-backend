# HairMe v20.1.6 변경사항

## 📅 업데이트 날짜: 2025-11-04

## 🎯 주요 변경사항

### ✅ 수직 얼굴 비율 데이터 수집 기능 추가

얼굴을 3등분(상안부/중안부/하안부)한 수직 비율 데이터를 DB에 저장하여, 향후 ML 모델 학습에 활용할 수 있도록 개선했습니다.

---

## 📊 새로 추가된 데이터

### DB 컬럼 (analysis_history 테이블)

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| `opencv_upper_face_ratio` | FLOAT | 상안부 높이 비율 (이마 영역, 0~1) |
| `opencv_middle_face_ratio` | FLOAT | 중안부 높이 비율 (눈~코 영역, 0~1) |
| `opencv_lower_face_ratio` | FLOAT | 하안부 높이 비율 (입~턱 영역, 0~1) |

**특징:**
- 카메라 거리에 관계없이 **비율**로 저장되어 일관성 보장
- OpenCV로 자동 측정 후 DB 저장
- ML 모델 학습 시 **얼굴 형태 판별의 정확도 향상** 기대

---

## 🔧 수정된 파일

### 1. `main.py`
- **AnalysisHistory 모델**: 수직 비율 컬럼 3개 추가 (line 265-268)
- **save_to_database 함수**: 수직 비율 저장 로직 추가 (line 744-747)
- **migrate_database_schema 함수**: 자동 마이그레이션에 수직 비율 컬럼 추가 (line 377-448)
- **버전 정보**: 20.1.4 → 20.1.6 업데이트

### 2. `db_schema_v20.1.6.sql` (신규 파일)
- 수동 마이그레이션 SQL 스크립트 작성
- 백업, 컬럼 추가, 검증 쿼리 포함

---

## 🚀 배포 방법

### 자동 마이그레이션 (권장)
서버 재시작 시 자동으로 DB 스키마 업데이트됩니다:
```bash
python main.py
```

### 수동 마이그레이션 (선택사항)
MySQL에서 직접 실행:
```bash
mysql -u admin -p hairme < db_schema_v20.1.6.sql
```

---

## 📈 데이터 수집 효과

### Before (v20.1.4)
```python
# 수평 비율만 저장
opencv_face_ratio: 1.23        # 얼굴 높이/너비
opencv_forehead_ratio: 0.85    # 이마 너비
opencv_cheekbone_ratio: 0.92   # 광대 너비
opencv_jaw_ratio: 0.78         # 턱 너비
```

### After (v20.1.6)
```python
# 수평 + 수직 비율 모두 저장
opencv_face_ratio: 1.23        # 얼굴 높이/너비
opencv_forehead_ratio: 0.85    # 이마 너비
opencv_cheekbone_ratio: 0.92   # 광대 너비
opencv_jaw_ratio: 0.78         # 턱 너비

# ✨ 새로 추가된 데이터
opencv_upper_face_ratio: 0.33   # 상안부 비율
opencv_middle_face_ratio: 0.33  # 중안부 비율
opencv_lower_face_ratio: 0.33   # 하안부 비율
```

---

## 🎓 ML 모델 학습 활용 방안

### 현재 데이터로 할 수 있는 것

1. **얼굴형 판별 정확도 향상**
   - 수평 비율만으로는 구분이 어려운 경우 (예: 둥근형 vs 각진형)
   - 수직 비율을 추가 특징으로 활용하여 정확도 개선

2. **개인 맞춤 추천**
   - 이마가 넓은 사람 → 시스루뱅 추천
   - 턱이 긴 사람 → 단발 추천
   - 수직 비율로 세밀한 추천 가능

3. **데이터 분석**
   ```sql
   -- 얼굴형별 평균 수직 비율 분석
   SELECT
       face_shape,
       AVG(opencv_upper_face_ratio) as avg_upper,
       AVG(opencv_middle_face_ratio) as avg_middle,
       AVG(opencv_lower_face_ratio) as avg_lower
   FROM analysis_history
   WHERE opencv_upper_face_ratio IS NOT NULL
   GROUP BY face_shape;
   ```

---

## ✅ 체크리스트

- [x] DB 스키마에 수직 비율 컬럼 추가
- [x] AnalysisHistory 모델 업데이트
- [x] save_to_database 함수 수정
- [x] 자동 마이그레이션 로직 추가
- [x] 버전 정보 업데이트 (20.1.6)
- [x] 문법 체크 (Python 컴파일 성공)
- [ ] 실제 서버 테스트 (배포 후 확인 필요)
- [ ] DB 데이터 수집 확인 (이미지 업로드 후 검증)

---

## 🔜 다음 단계

### 즉시 가능:
1. **데이터 수집 시작**
   - 서버 재시작 후 사용자 이미지 업로드 시 자동으로 수직 비율 수집
   - 1-2주 후 충분한 데이터 확보 가능

2. **데이터 검증**
   ```sql
   -- 수직 비율 데이터 수집률 확인
   SELECT
       COUNT(*) as total,
       SUM(CASE WHEN opencv_upper_face_ratio IS NOT NULL THEN 1 ELSE 0 END) as collected,
       ROUND(SUM(CASE WHEN opencv_upper_face_ratio IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as coverage
   FROM analysis_history;
   ```

### 중기 계획:
3. **DB 데이터 → CSV 추출 스크립트 작성**
   - analysis_history 테이블 → 학습용 CSV 변환
   - 얼굴 비율 + 피드백 데이터 통합

4. **실제 데이터로 ML 모델 재학습**
   - 합성 데이터 + 실제 데이터 혼합 학습
   - 성능 비교 및 개선

---

## 📞 문의사항

- 배포 이슈: GitHub Issues
- 데이터 분석 문의: 개발팀

---

**작성자**: Claude Code
**검토 필요**: DB 백업 후 배포 권장
