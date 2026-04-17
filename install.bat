@echo off
chcp 65001 >nul
echo ════════════════════════════════════════════════
echo   RuVoiCer — установка
echo ════════════════════════════════════════════════
echo.

:: ── Python 3.11 (через winget) ─────────────────
echo Проверка Python 3.11...
py -3.11 -c "import sys; print(f'Python {sys.version}')" >nul 2>&1
if errorlevel 1 (
    :: py launcher может вернуть 0 даже без рантайма — проверяем доп. способом
    goto :py_missing
)
:: Дополнительная проверка: реально ли работает
py -3.11 -c "print('OK')" 2>nul | findstr /c:"OK" >nul
if errorlevel 1 goto :py_missing
goto :py_ok

:py_missing
echo Python 3.11 не найден. Установка через winget...
winget install --id Python.Python.3.11 -e --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo [ОШИБКА] Не удалось установить Python 3.11.
    echo Установите вручную: winget install Python.Python.3.11
    pause
    exit /b 1
)
echo.
echo Python 3.11 установлен.
echo [!] Перезапустите терминал и запустите install.bat повторно,
echo     чтобы PATH обновился.
pause
exit /b 0

:py_ok
for /f "tokens=*" %%V in ('py -3.11 --version 2^>nul') do echo %%V найден.

:: ── Виртуальное окружение ───────────────────────
if not exist "venv\" (
    echo Создание виртуального окружения...
    py -3.11 -m venv venv
)
call venv\Scripts\activate.bat

:: ── FFmpeg (через winget) ───────────────────────
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo FFmpeg не найден. Установка через winget...
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [ВНИМАНИЕ] Не удалось установить FFmpeg через winget.
        echo Установите вручную: winget install Gyan.FFmpeg
    ) else (
        echo FFmpeg установлен.
    )
) else (
    echo FFmpeg найден.
)

:: ── MSVC Build Tools (через winget, нужен для сборки TTS) ──
echo.
where cl >nul 2>&1
if errorlevel 1 (
    echo C++ компилятор не найден. Установка MSVC Build Tools через winget...
    echo Это нужно для сборки пакета TTS. Загрузка ~2 ГБ.
    winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-source-agreements --accept-package-agreements --override "--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended --quiet --wait"
    if errorlevel 1 (
        echo [ВНИМАНИЕ] Не удалось установить MSVC Build Tools.
        echo Установите вручную: https://visualstudio.microsoft.com/visual-cpp-build-tools/
    ) else (
        echo MSVC Build Tools установлены.
    )
) else (
    echo C++ компилятор найден.
)

:: ── Активация MSVC (если не в PATH) ────────────
where cl >nul 2>&1
if errorlevel 1 (
    for /f "usebackq tokens=*" %%i in (`powershell -Command "(& 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe' -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null)"`) do set "VSINSTALL=%%i"
    if defined VSINSTALL (
        call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
        echo MSVC компилятор активирован.
    )
)

:: ── PyTorch + CUDA ──────────────────────────────
echo.
echo Установка PyTorch с поддержкой CUDA 12.1...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

:: ── Сборочные зависимости для TTS ──────────────
echo.
echo Установка сборочных зависимостей...
pip install numpy>=1.24.0 Cython

:: ── Зависимости ─────────────────────────────────
echo.
echo Установка зависимостей...
pip install -r requirements.txt

:: ── Готово ──────────────────────────────────────
echo.
echo ════════════════════════════════════════════════
echo   Установка завершена!
echo   Запуск: run.bat
echo ════════════════════════════════════════════════
pause
