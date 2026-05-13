@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python не найден в PATH.
    echo Установите Python и убедитесь, что команда python доступна в консоли.
    pause
    exit /b 1
)

python "%~dp0yolo_labeler.py"
if errorlevel 1 (
    echo.
    echo Программа завершилась с ошибкой.
    echo Если не установлен Pillow, выполните: pip install pillow
)

