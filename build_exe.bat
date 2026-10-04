@echo off
rem Сборка одного .exe-файла (результат: dist\VoiceProcessor.exe)
pip install -r requirements.txt pyinstaller
pyinstaller --noconsole --onefile --name VoiceProcessor voice_app.py
pause
