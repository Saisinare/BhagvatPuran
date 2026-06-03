@echo off
echo ========================================
echo   Starting RAGita Frontend
echo ========================================
echo.

cd frontend

echo Opening RAGita in your default browser...
echo Frontend URL: http://localhost:8080
echo.

echo Starting local server...
python -m http.server 8080
