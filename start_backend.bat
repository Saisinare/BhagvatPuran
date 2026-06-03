@echo off
echo ========================================
echo   Starting RAGita Backend Server
echo ========================================
echo.

cd backend

echo Checking Python environment...
python --version
echo.

echo Installing/updating dependencies...
pip install -r requirements.txt
echo.

echo Starting FastAPI server on http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo.
python main.py
