@echo off
REM Creeaza mediul virtual .venv si instaleaza dependentele (o singura data).
cd /d "%~dp0"

set PY=python
where py >nul 2>nul
if %errorlevel%==0 set PY=py -3

echo [1/3] Creez mediul virtual .venv ...
%PY% -m venv .venv
if errorlevel 1 (
  echo Nu am gasit Python. Instaleaza Python 3.10+ de pe python.org ^(bifeaza "Add to PATH"^).
  pause
  exit /b 1
)

echo [2/3] Actualizez pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/3] Instalez dependentele (torch poate dura cateva minute) ...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo Instalarea a esuat. Verifica conexiunea la internet si incearca din nou.
  pause
  exit /b 1
)

echo.
echo Gata. Porneste interfata cu run_gui.bat
pause
