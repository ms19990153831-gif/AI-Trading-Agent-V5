@echo off
cd /d "%~dp0"
if not exist "main.py" cd /d "C:\Users\Administrator\Documents\Codex\2026-08-21\new-chat-2\outputs\AI_TRADING_AGENT"
python main.py --mode demo --cycles 12
pause
