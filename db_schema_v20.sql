-- HairMe v20 Schema Update
-- 피드백 기능을 위한 컬럼 추가

USE hairme;

-- 1. 추천 스타일 저장 컬럼
ALTER TABLE analysis_history
ADD COLUMN recommended_styles JSON COMMENT '추천된 3개 헤어스타일 (JSON 배열)';

-- 2. 각 스타일별 피드백 컬럼
ALTER TABLE analysis_history
ADD COLUMN style_1_feedback ENUM('like', 'dislike') DEFAULT NULL COMMENT '첫 번째 스타일 피드백',
ADD COLUMN style_2_feedback ENUM('like', 'dislike') DEFAULT NULL COMMENT '두 번째 스타일 피드백',
ADD COLUMN style_3_feedback ENUM('like', 'dislike') DEFAULT NULL COMMENT '세 번째 스타일 피드백';

-- 3. 네이버 이미지 검색 클릭 여부
ALTER TABLE analysis_history
ADD COLUMN style_1_naver_clicked BOOLEAN DEFAULT FALSE COMMENT '첫 번째 스타일 네이버 클릭',
ADD COLUMN style_2_naver_clicked BOOLEAN DEFAULT FALSE COMMENT '두 번째 스타일 네이버 클릭',
ADD COLUMN style_3_naver_clicked BOOLEAN DEFAULT FALSE COMMENT '세 번째 스타일 네이버 클릭';

-- 4. 피드백 제출 시각
ALTER TABLE analysis_history
ADD COLUMN feedback_at DATETIME DEFAULT NULL COMMENT '피드백 제출 시각';

-- 확인
DESCRIBE analysis_history;
