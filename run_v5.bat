@echo off
cd /d "%~dp0"
if not exist "data" mkdir "data"
if not exist "data\logs" mkdir "data\logs"
set "LOGFILE=%~dp0data\logs\bot_%date:~0,4%%date:~5,2%%date:~8,2%.log"
set "PID="
if exist "data\trader.pid" set /p PID=<"data\trader.pid"
if defined PID (
  tasklist /FI "PID eq %PID%" 2>nul | find "%PID%" >nul
  if not errorlevel 1 (
    echo AI is already running as PID %PID%.
    echo Close the AI window to stop it. Do not start a second copy.
    pause
    exit /b 0
  )
)
(echo %LOGFILE%)> "data\current_bot_log.txt"
echo [%date% %time%] AI bot starting...
set "PYTHONIOENCODING=utf-8"
python -u main.py --mode v5 --cycles 288 --interval 300 >> "%LOGFILE%" 2>&1
echo.
echo Bot exited with code %errorlevel%. Log: %LOGFILE%
timeout /t 3 /nobreak >nul
