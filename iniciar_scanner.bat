@echo off
title S&P 500 RSI Scanner
cd /d "%~dp0"

echo ============================================
echo   S^&P 500 RSI Scanner
echo ============================================
echo.

echo Actualizando codigo...
git pull origin claude/zealous-pascal-Ysg2j

echo.
echo Iniciando scanner (Ctrl+C para detener)...
echo.

python sp500_rsi_scanner.py --top 50 --watch

echo.
echo Scanner detenido.
pause
