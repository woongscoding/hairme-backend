@echo off
REM 하루 한 번 광고 경로를 점검하고 결과를 로그에 덧붙인다 (작업 스케줄러용).
REM PYTHONIOENCODING: 리디렉션 시 Python이 cp949로 쓰면 로그가 깨진다.
set PYTHONIOENCODING=utf-8
cd /d C:\Users\pcw99\Hairstyle_server
python scripts\admob_watch.py --days 2 >> "%~dp0.admob_watch.log" 2>&1
echo. >> "%~dp0.admob_watch.log"
