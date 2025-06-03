@echo off
echo Activating Conda Environment...
call conda activate kopyae_1

echo Starting FastAPI backend...
start cmd /k "cd backend && uvicorn main:app --reload"

timeout /t 5 > nul

echo Opening frontend site in browser...
start frontend\index.html
