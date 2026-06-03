@echo off
echo ========================================
echo   RAGita - Starting Full Stack
echo ========================================
echo.

echo Starting Backend on port 8000...
start cmd /k "cd /d %~dp0 && start_backend.bat"

timeout /t 3 /nobreak > nul

echo Starting Frontend on port 8080...
start cmd /k "cd /d %~dp0 && start_frontend.bat"

timeout /t 2 /nobreak > nul

echo.
echo ========================================
echo   RAGita is now running!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Frontend: http://localhost:8080
echo.
echo Opening frontend in browser...
timeout /t 3 /nobreak > nul
start http://localhost:8080

echo.
echo Press any key to stop all services...
pause > nul

taskkill /FI "WindowTitle eq *start_backend.bat*" /F
taskkill /FI "WindowTitle eq *start_frontend.bat*" /F
