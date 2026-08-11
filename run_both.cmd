@echo off
pushd "%~dp0"

if "%GROQ_API_KEY%"=="" echo WARNING: GROQ_API_KEY is not set. AI features will use the non-AI fallback.

echo Starting HTML/CSS interface on http://127.0.0.1:8000
start "HTML interface" cmd /k python serve_frontend.py

timeout /t 3 /nobreak > nul

echo Starting Streamlit interface on http://127.0.0.1:8501
start "Streamlit interface" cmd /k streamlit run app.py --server.port=8501

timeout /t 5 /nobreak > nul
start "" "http://127.0.0.1:8000"

echo Both interfaces started. Close their windows to stop them.
popd
