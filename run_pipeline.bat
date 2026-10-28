@echo off
REM Ruleaza tot pipeline-ul din linia de comanda: date -> antrenare -> evaluare.
REM Atentie: suprascrie data\dataset.npz si models\unet.pt
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Mediul virtual nu exista. Ruleaza intai setup.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m lt.dataset --n 2000 || goto :err
.venv\Scripts\python.exe -m lt.train --epochs 60 || goto :err
.venv\Scripts\python.exe -m lt.evaluate || goto :err
echo Gata. Rapoartele sunt in reports\unet
pause
exit /b 0
:err
echo A aparut o eroare.
pause
exit /b 1
