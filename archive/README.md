# Archive 폴더

이 폴더는 2025-11-04에 프로젝트 정리 과정에서 생성되었습니다.

## 📦 보관된 파일들

### 목적
- 더 이상 사용하지 않지만, 참고용으로 보관할 파일들
- 필요시 복원 가능
- 일정 기간 후 안전하게 삭제 가능

---

## 📂 폴더 구조

### 1. `old_versions/` - 백업 코드
- `main_v19_backup.py` - v20 업데이트 전 백업

### 2. `old_scripts/` - 구버전 스크립트 (3개)
- `database.py` - DB 연결 설정 (main.py에 통합됨)
- `models.py` - SQLAlchemy 모델 (main.py에 통합됨)
- `init_db.py` - 초기 DB 설정 스크립트 (더 이상 사용 안 함)

### 3. `old_schemas/` - 구버전 스키마 (3개)
- `db_schema_v20.sql` - v20 초기 스키마
- `db_schema_v20.1.5.sql` - v20.1.5 스키마
- `update_schema.py` - 수동 스키마 업데이트 스크립트

**최신 버전**: `db_schema_v20.1.6.sql` (루트 디렉토리)

### 4. `test_files/` - 테스트 관련 (8개)
- `test_api.py` - API 테스트 스크립트
- `fortest.py` - 기타 테스트 코드
- `test_face.jpg` - 테스트 이미지 1 (45KB)
- `test_face2.jpg` - 테스트 이미지 2 (18KB)
- `test_api_v20.sh` - v20 API 테스트 쉘 스크립트
- `logs-v9.json` - 오래된 로그 파일 (33KB)
- `task-info.txt` - 빈 파일

### 5. `old_task_definitions/` - ECS Task Definitions (31개)
오래된 AWS ECS task definition 파일들
- v2 ~ v10
- v15 ~ v19
- v20.1.1 ~ v20.1.5
- ML 관련 중간 버전들

**현재 사용 중**: `task-def-v20.1-ml-final.json` (task_def/ 폴더)

---

## 📊 정리 결과

### Before (정리 전)
- 루트 디렉토리: **54개 파일**
- task_def 폴더: **23개 파일**

### After (정리 후)
- 루트 디렉토리: **15개 파일** (필수 파일만 유지)
- task_def 폴더: **1개 파일** (최신 버전만)
- archive 폴더: **39개 파일** (백업 보관)

---

## 🗑️ 완전 삭제 시기

다음 조건이 충족되면 archive 폴더 전체 삭제 가능:

1. ✅ 프로덕션 서버가 v20.1.6으로 안정적으로 운영 중
2. ✅ 최소 1개월 이상 문제 없이 운영됨
3. ✅ Git에 모든 변경사항이 커밋되어 있음
4. ✅ 롤백할 필요가 없다고 판단됨

**권장 삭제 시기**: 2025년 12월 이후

---

## 🔄 파일 복원 방법

필요한 파일이 있다면:

```bash
# archive 폴더에서 루트로 복사
cp archive/old_scripts/database.py .

# 또는 이동
mv archive/test_files/test_face.jpg .
```

---

**정리 날짜**: 2025-11-04
**정리자**: Claude Code
**프로젝트 버전**: v20.1.6
