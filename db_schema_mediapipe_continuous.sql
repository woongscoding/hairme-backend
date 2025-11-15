-- ============================================================
-- HairMe MediaPipe Continuous Features Migration
-- MediaPipe 연속형 변수 저장용 컬럼 추가
-- ============================================================

USE hairme;

-- ========== 1. 백업 (안전장치) ==========
CREATE TABLE IF NOT EXISTS analysis_history_backup_mediapipe AS
SELECT * FROM analysis_history;

SELECT COUNT(*) as backup_count FROM analysis_history_backup_mediapipe;

-- ========== 2. MediaPipe 연속형 변수 컬럼 추가 ==========

ALTER TABLE analysis_history
-- 얼굴 측정값 (6개)
ADD COLUMN IF NOT EXISTS mediapipe_face_ratio FLOAT DEFAULT NULL COMMENT '얼굴 비율 (높이/너비) - 0.8~1.6',
ADD COLUMN IF NOT EXISTS mediapipe_forehead_width FLOAT DEFAULT NULL COMMENT '이마 너비 (픽셀) - 100~300',
ADD COLUMN IF NOT EXISTS mediapipe_cheekbone_width FLOAT DEFAULT NULL COMMENT '광대 너비 (픽셀) - 120~350',
ADD COLUMN IF NOT EXISTS mediapipe_jaw_width FLOAT DEFAULT NULL COMMENT '턱 너비 (픽셀) - 80~280',
ADD COLUMN IF NOT EXISTS mediapipe_forehead_ratio FLOAT DEFAULT NULL COMMENT '이마/광대 비율 - 0.7~1.2',
ADD COLUMN IF NOT EXISTS mediapipe_jaw_ratio FLOAT DEFAULT NULL COMMENT '턱/광대 비율 - 0.6~1.1',

-- 피부 측정값 (2개 + 예비)
ADD COLUMN IF NOT EXISTS mediapipe_ITA_value FLOAT DEFAULT NULL COMMENT 'ITA 피부톤 값 (-50~70도)',
ADD COLUMN IF NOT EXISTS mediapipe_hue_value FLOAT DEFAULT NULL COMMENT 'HSV Hue 값 (0~179)',

-- MediaPipe 메타데이터
ADD COLUMN IF NOT EXISTS mediapipe_confidence FLOAT DEFAULT NULL COMMENT 'MediaPipe 신뢰도 (0.0~1.0)',
ADD COLUMN IF NOT EXISTS mediapipe_features_complete BOOLEAN DEFAULT FALSE COMMENT '연속형 변수 완전성 플래그';

-- ========== 3. 인덱스 추가 (성능 최적화) ==========
-- MediaPipe 데이터가 있는 레코드 빠른 조회
CREATE INDEX IF NOT EXISTS idx_mediapipe_complete
ON analysis_history(mediapipe_features_complete)
WHERE mediapipe_features_complete = TRUE;

-- 피드백이 있는 학습 데이터 조회
CREATE INDEX IF NOT EXISTS idx_training_data
ON analysis_history(mediapipe_features_complete, style_1_feedback)
WHERE mediapipe_features_complete = TRUE AND style_1_feedback IS NOT NULL;

-- ========== 4. 확인 ==========
DESCRIBE analysis_history;

-- MediaPipe 컬럼 확인
SELECT
    COLUMN_NAME,
    COLUMN_TYPE,
    COLUMN_COMMENT
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'hairme'
  AND TABLE_NAME = 'analysis_history'
  AND COLUMN_NAME LIKE 'mediapipe%'
ORDER BY ORDINAL_POSITION;

-- ========== 5. 데이터 검증 쿼리 ==========
-- 최근 10개 레코드의 MediaPipe 데이터 확인
SELECT
    id,
    face_shape,
    personal_color,
    mediapipe_face_ratio,
    mediapipe_forehead_width,
    mediapipe_cheekbone_width,
    mediapipe_jaw_width,
    mediapipe_forehead_ratio,
    mediapipe_jaw_ratio,
    mediapipe_ITA_value,
    mediapipe_hue_value,
    mediapipe_confidence,
    mediapipe_features_complete,
    created_at
FROM analysis_history
ORDER BY id DESC
LIMIT 10;

-- ========== 6. 통계 쿼리 ==========
-- MediaPipe 데이터 수집률
SELECT
    COUNT(*) as total_records,
    SUM(CASE WHEN mediapipe_features_complete = TRUE THEN 1 ELSE 0 END) as mediapipe_complete,
    ROUND(
        SUM(CASE WHEN mediapipe_features_complete = TRUE THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as mediapipe_coverage_percent,
    SUM(CASE WHEN style_1_feedback IS NOT NULL THEN 1 ELSE 0 END) as feedback_count,
    SUM(CASE
        WHEN mediapipe_features_complete = TRUE AND style_1_feedback IS NOT NULL
        THEN 1 ELSE 0
    END) as training_data_ready
FROM analysis_history;

-- ========== 7. 학습 데이터 샘플 확인 ==========
-- 피드백이 있는 MediaPipe 데이터 (학습 가능한 레코드)
SELECT
    id,
    face_shape,
    personal_color,
    mediapipe_face_ratio,
    mediapipe_ITA_value,
    JSON_EXTRACT(recommended_styles, '$[0].style_name') as style_1_name,
    style_1_feedback,
    style_1_naver_clicked,
    created_at
FROM analysis_history
WHERE mediapipe_features_complete = TRUE
  AND style_1_feedback IS NOT NULL
ORDER BY created_at DESC
LIMIT 5;

-- ========== 8. 롤백 방법 (문제 발생 시) ==========
/*
-- MediaPipe 컬럼 제거
ALTER TABLE analysis_history
DROP COLUMN mediapipe_face_ratio,
DROP COLUMN mediapipe_forehead_width,
DROP COLUMN mediapipe_cheekbone_width,
DROP COLUMN mediapipe_jaw_width,
DROP COLUMN mediapipe_forehead_ratio,
DROP COLUMN mediapipe_jaw_ratio,
DROP COLUMN mediapipe_ITA_value,
DROP COLUMN mediapipe_hue_value,
DROP COLUMN mediapipe_confidence,
DROP COLUMN mediapipe_features_complete;

-- 또는 전체 복원
DROP TABLE analysis_history;
RENAME TABLE analysis_history_backup_mediapipe TO analysis_history;
*/

-- ========== 완료 메시지 ==========
SELECT '✅ MediaPipe continuous features migration completed!' as status;
SELECT 'Added 10 columns for continuous ML training' as details;
SELECT 'Next step: Update save_to_database() to store MediaPipe features' as action;
