#!/bin/bash
# ============================================================
# 1000개 AI 얼굴 데이터 수집 스크립트
# 목표: 6000개 학습 샘플 (1000 × 6)
# 예상 시간: ~35분
# ============================================================

set -e  # 에러 발생 시 중단

echo "============================================================"
echo "🚀 대규모 AI 얼굴 데이터 수집 시작"
echo "============================================================"
echo "목표: 1000개 AI 얼굴"
echo "예상 샘플: ~6000개 (얼굴당 6개)"
echo "예상 시간: ~35분 (delay=2.0 기준)"
echo "============================================================"
echo ""

# API 키 확인
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ 에러: GEMINI_API_KEY 환경변수가 설정되지 않았습니다."
    echo "실행 방법: export GEMINI_API_KEY='your-api-key'"
    exit 1
fi

# 출력 디렉토리 생성
mkdir -p data_source
mkdir -p logs

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/collect_${TIMESTAMP}.log"

echo "📝 로그 파일: $LOG_FILE"
echo ""

# ============================================================
# 옵션 1: 한 번에 1000개 수집 (Gemini 무료 티어 주의!)
# ============================================================

echo "⚠️ Gemini 무료 티어 제한:"
echo "  - 1500 requests/day"
echo "  - 60 requests/minute"
echo ""
echo "1000개 AI 얼굴 = ~2000 API 호출"
echo "→ 무료 티어 초과 가능성 있음!"
echo ""
read -p "계속하시겠습니까? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "취소되었습니다."
    exit 0
fi

echo ""
echo "============================================================"
echo "🔄 데이터 수집 시작..."
echo "============================================================"
echo ""

# Python 스크립트 실행
python scripts/collect_ai_face_training_data.py \
    -n 1000 \
    --delay 2.0 \
    -o "data_source/ai_face_1000_${TIMESTAMP}.npz" \
    2>&1 | tee "$LOG_FILE"

# 결과 확인
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ 수집 완료!"
    echo "============================================================"

    # 통계 출력
    python -c "
import numpy as np
import sys

try:
    data = np.load('data_source/ai_face_1000_${TIMESTAMP}.npz', allow_pickle=True)

    print(f'\n📊 최종 통계:')
    print(f'  - 총 샘플: {len(data[\"scores\"])}개')
    print(f'  - 추천 (≥70): {(data[\"scores\"] >= 70).sum()}개')
    print(f'  - 비추천 (<70): {(data[\"scores\"] < 70).sum()}개')

    # 얼굴형 분포
    from collections import Counter
    metadata = data['metadata']
    face_shapes = [m['face_shape'] for m in metadata]
    print(f'\n📊 얼굴형 분포:')
    for shape, count in Counter(face_shapes).most_common():
        print(f'  - {shape}: {count}개')

    # 피부톤 분포
    skin_tones = [m['skin_tone'] for m in metadata]
    print(f'\n📊 피부톤 분포:')
    for tone, count in Counter(skin_tones).most_common():
        print(f'  - {tone}: {count}개')

    # 점수 통계
    scores = data['scores']
    print(f'\n📊 점수 통계:')
    print(f'  - 평균: {scores.mean():.1f}')
    print(f'  - 표준편차: {scores.std():.1f}')
    print(f'  - 최소: {scores.min():.1f}')
    print(f'  - 최대: {scores.max():.1f}')

except Exception as e:
    print(f'❌ 통계 출력 실패: {e}')
    sys.exit(1)
"

    echo ""
    echo "============================================================"
    echo "🎉 완료! 이제 모델을 학습하세요:"
    echo "  python scripts/train_model_v4.py --data data_source/ai_face_1000_${TIMESTAMP}.npz"
    echo "============================================================"
else
    echo ""
    echo "❌ 수집 중 오류가 발생했습니다."
    echo "로그 파일을 확인하세요: $LOG_FILE"
    exit 1
fi
