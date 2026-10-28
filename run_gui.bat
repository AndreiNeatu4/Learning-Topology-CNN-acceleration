@echo off
REM Porneste interfata grafica in browser (http://localhost:8501).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Mediul virtual nu exista. Ruleaza intai setup.bat
  pause
  exit /b 1
)
start "" cmd /c "timeout /t 4 >nul & start http://localhost:8501"
.venv\Scripts\python.exe -m streamlit run app.py
pause
