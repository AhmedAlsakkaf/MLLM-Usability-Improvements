@echo off
rem Starts the UI Analysis Lab in your browser (http://localhost:8501).
rem Close this window to stop the app.
cd /d "%~dp0"
python -m streamlit run evaluation\app.py --browser.gatherUsageStats false
pause
