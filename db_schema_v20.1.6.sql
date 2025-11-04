-- HairMe v20.1.6 Schema Migration
-- 수직 비율 데이터 추가 (얼굴 3등분 비율)

USE hairme;

-- ========== 1. 백업 (안전장치) ==========
CREATE TABLE IF NOT EXISTS analysis_history_backup_v20_1_6 AS
SELECT * FROM analysis_history;

-- 확인
SELECT COUNT(*) as backup_count FROM analysis_history_backup_v20_1_6;

-- ========== 2. 수직 비율 컬럼 추가 ==========
-- 얼굴을 3등분한 각 영역의 높이 비율

ALTER TABLE analysis_history
ADD COLUMN IF NOT EXISTS opencv_upper_face_ratio FLOAT DEFAULT NULL COMMENT '상안부 높이 비율 (이마 영역)',
ADD COLUMN IF NOT EXISTS opencv_middle_face_ratio FLOAT DEFAULT NULL COMMENT '중안부 높이 비율 (눈~코 영역)',
ADD COLUMN IF NOT EXISTS opencv_lower_face_ratio FLOAT DEFAULT NULL COMMENT '하안부 높이 비율 (입~턱 영역)';

-- ========== 3. 확인 ==========
DESCRIBE analysis_history;

-- 새로 추가된 컬럼이 표시되어야 함:
-- - opencv_upper_face_ratio
-- - opencv_middle_face_ratio
-- - opencv_lower_face_ratio

-- ========== 4. 데이터 검증 ==========
-- 최근 10개 레코드의 OpenCV 데이터 확인
SELECT
    id,
    opencv_face_ratio,
    opencv_forehead_ratio,
    opencv_cheekbone_ratio,
    opencv_jaw_ratio,
    opencv_upper_face_ratio,
    opencv_middle_face_ratio,
    opencv_lower_face_ratio,
    opencv_prediction,
    opencv_confidence,
    created_at
FROM analysis_history
ORDER BY id DESC
LIMIT 10;

-- ========== 5. 인덱스 추가 (성능 최적화) ==========
-- OpenCV 데이터가 있는 레코드만 빠르게 조회하기 위한 인덱스
CREATE INDEX IF NOT EXISTS idx_opencv_data_exists
ON analysis_history(opencv_confidence)
WHERE opencv_confidence IS NOT NULL;

-- ========== 6. 통계 쿼리 ==========
-- OpenCV 데이터 수집률 확인
SELECT
    COUNT(*) as total_records,
    SUM(CASE WHEN opencv_face_ratio IS NOT NULL THEN 1 ELSE 0 END) as opencv_analyzed,
    ROUND(
        SUM(CASE WHEN opencv_face_ratio IS NOT NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as opencv_coverage_percent,
    SUM(CASE WHEN opencv_upper_face_ratio IS NOT NULL THEN 1 ELSE 0 END) as vertical_ratio_collected
FROM analysis_history;

-- ========== 7. 롤백 방법 (문제 발생 시) ==========
-- 백업 테이블에서 복원
/*
-- 컬럼 제거 (롤백 시 실행)
ALTER TABLE analysis_history
DROP COLUMN opencv_upper_face_ratio,
DROP COLUMN opencv_middle_face_ratio,
DROP COLUMN opencv_lower_face_ratio;

-- 또는 전체 복원
DROP TABLE analysis_history;
RENAME TABLE analysis_history_backup_v20_1_6 TO analysis_history;
*/

-- ========== 완료 메시지 ==========
SELECT '✅ Schema migration v20.1.6 completed successfully!' as status;
SELECT 'Added columns: opencv_upper_face_ratio, opencv_middle_face_ratio, opencv_lower_face_ratio' as details;
