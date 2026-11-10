@echo off
echo ==============================================================================
echo Launching Hindi Text-to-Speech & AI Story Narration Studio...
echo ==============================================================================
cd /d "%~dp0"
streamlit run app.py --server.port 8501
pause

