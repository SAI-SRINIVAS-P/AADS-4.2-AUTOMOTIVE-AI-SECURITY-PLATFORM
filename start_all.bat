@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
start "AADS Dashboard" /D "%~dp0" cmd /k ".venv\Scripts\python.exe -m streamlit run app.py"
start "AADS API" /D "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn api:app --reload"
