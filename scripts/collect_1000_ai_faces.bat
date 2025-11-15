@echo off
REM ============================================================
REM 1000개 AI 얼굴 데이터 수집 스크립트 (Windows)
REM 목표: 6000개 학습 샘플 (1000 × 6)
REM 예상 시간: ~35분
REM ============================================================

echo ============================================================
echo 🚀 대규모 AI 얼굴 데이터 수집 시작
echo ============================================================
echo 목표: 1000개 AI 얼굴
echo 예상 샘플: ~6000개 (얼굴당 6개)
echo 예상 시간: ~35분 (delay=2.0 기준)
echo ============================================================
echo.

REM API 키 확인
if "%GEMINI_API_KEY%"=="" (
    echo ❌ 에러: GEMINI_API_KEY 환경변수가 설정되지 않았습니다.
    echo 실행 방법: set GEMINI_API_KEY=your-api-key
    pause
    exit /b 1
)

REM 출력 디렉토리 생성
if not exist "data_source" mkdir data_source
if not exist "logs" mkdir logs

REM 타임스탬프
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do (set mytime=%%a%%b)
set TIMESTAMP=%mydate%_%mytime%
set LOG_FILE=logs\collect_%TIMESTAMP%.log

echo 📝 로그 파일: %LOG_FILE%
echo.

REM ============================================================
REM Gemini API 제한 경고
REM ============================================================

echo ⚠️ Gemini 무료 티어 제한:
echo   - 1500 requests/day
echo   - 60 requests/minute
echo.
echo 1000개 AI 얼굴 = ~2000 API 호출
echo → 2일에 걸쳐 수집 권장!
echo.
echo 계속하시겠습니까?
pause

echo.
echo ============================================================
echo 🔄 데이터 수집 시작...
echo ============================================================
echo.

REM Python 스크립트 실행
python scripts\collect_ai_face_training_data.py -n 1000 --delay 2.0 -o "data_source\ai_face_1000_%TIMESTAMP%.npz" 2>&1 | tee %LOG_FILE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo ✅ 수집 완료!
    echo ============================================================

    REM 통계 출력
    python -c "import numpy as np; data = np.load('data_source/ai_face_1000_%TIMESTAMP%.npz', allow_pickle=True); print(f'\n총 샘플: {len(data[\"scores\"])}개')"

    echo.
    echo ============================================================
    echo 🎉 완료! 이제 모델을 학습하세요:
    echo   python scripts/train_model_v4.py --data data_source/ai_face_1000_%TIMESTAMP%.npz
    echo ============================================================
) else (
    echo.
    echo ❌ 수집 중 오류가 발생했습니다.
    echo 로그 파일을 확인하세요: %LOG_FILE%
)

pause
