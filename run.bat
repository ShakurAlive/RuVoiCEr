@echo off
chcp 65001 >nul
call venv\Scripts\activate.bat
set COQUI_TOS_AGREED=1
set HF_HUB_ENABLE_HF_TRANSFER=0
set HF_HUB_DISABLE_XET=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set OMP_NUM_THREADS=1
set CT2_VERBOSE=-1
echo Запуск RuVoiCer...
python app.py 2> crash.log
echo.
echo ── Лог ошибок (crash.log): ──
type crash.log
pause
