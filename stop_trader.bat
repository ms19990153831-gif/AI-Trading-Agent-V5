@echo off
cd /d "%~dp0"
if not exist "data" cd /d "C:\Users\Administrator\Documents\Codex\2026-08-21\new-chat-2\outputs\AI_TRADING_AGENT"
set "PID="
if exist "%~dp0data\trader.pid" set /p PID=<"%~dp0data\trader.pid"
if defined PID (
  taskkill /F /PID %PID%
  echo Trader process %PID% stopped.
) else (
  echo No trader PID file found.
)
pause
