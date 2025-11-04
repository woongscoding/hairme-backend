-- HairMe v20.1.5 Schema Migration
-- 피드백 컬럼을 ENUM에서 VARCHAR로 변경

USE hairme;

-- ========== 1. 백업 (안전장치) ==========
CREATE TABLE IF NOT EXISTS analysis_history_backup_20241102 AS
SELECT * FROM analysis_history;

-- 확인
SELECT COUNT(*) as backup_count FROM analysis_history_backup_20241102;

-- ========== 2. 컬럼 타입 변경 (ENUM → VARCHAR) ==========
-- MySQL ENUM을 VARCHAR로 변경하면 기존 데이터는 자동으로 문자열로 변환됨

ALTER TABLE analysis_history
MODIFY COLUMN style_1_feedback VARCHAR(10) DEFAULT NULL COMMENT '첫 번째 스타일 피드백',
MODIFY COLUMN style_2_feedback VARCHAR(10) DEFAULT NULL COMMENT '두 번째 스타일 피드백',
MODIFY COLUMN style_3_feedback VARCHAR(10) DEFAULT NULL COMMENT '세 번째 스타일 피드백';

-- ========== 3. 확인 ==========
DESCRIBE analysis_history;

-- style_1_feedback, style_2_feedback, style_3_feedback이
-- varchar(10)로 표시되어야 함

-- ========== 4. 데이터 검증 ==========
-- 기존 피드백 데이터가 정상적으로 변환되었는지 확인
SELECT
    id,
    style_1_feedback,
    style_2_feedback,
    style_3_feedback,
    feedback_at
FROM analysis_history
WHERE style_1_feedback IS NOT NULL
   OR style_2_feedback IS NOT NULL
   OR style_3_feedback IS NOT NULL
ORDER BY id DESC
LIMIT 10;

-- ========== 5. 롤백 방법 (문제 발생 시) ==========
-- 백업 테이블에서 복원
/*
DROP TABLE analysis_history;
RENAME TABLE analysis_history_backup_20241102 TO analysis_history;
*/

-- ========== 완료 메시지 ==========
SELECT '✅ Schema migration completed successfully!' as status;