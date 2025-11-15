@echo off
REM ============================================================
REM AI 얼굴 데이터 수집 시작 (로그 기록)
REM ============================================================

echo ============================================================
echo 🚀 1000개 AI 얼굴 데이터 수집 시작
echo ============================================================
echo.

REM 타임스탬프 생성
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do (set mytime=%%a%%b)
set TIMESTAMP=%mydate%_%mytime%

REM 로그 디렉토리 생성
if not exist "logs" mkdir logs
set LOG_FILE=logs\collection_%TIMESTAMP%.log

echo 📝 로그 파일: %LOG_FILE%
echo 📊 실시간 모니터링: tail -f %LOG_FILE%
echo.

REM API 키 설정
set GEMINI_API_KEY=AIzaSyAn3qJqKYj-NCbdwIQsgJWOwr90NjTKG1U

REM 수집 시작 (로그 저장)
echo [%date% %time%] 데이터 수집 시작 >> %LOG_FILE%
echo ============================================================ >> %LOG_FILE%
echo 🚀 1000개 AI 얼굴 데이터 수집 >> %LOG_FILE%
echo ============================================================ >> %LOG_FILE%
echo. >> %LOG_FILE%

python scripts\collect_ai_face_training_data.py -n 1000 --delay 2.0 -o data_source\ai_face_1000.npz 2>&1 | tee %LOG_FILE%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo ✅ 수집 완료!
    echo ============================================================
    echo 로그 파일: %LOG_FILE%
    echo.
) else (
    echo.
    echo ❌ 오류 발생! 로그 확인: %LOG_FILE%
    echo.
)

pause
