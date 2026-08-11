#!/usr/bin/env bash
# Runs both interfaces at the same time on different ports.
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

if [ -z "$GROQ_API_KEY" ]; then
  echo "WARNING: GROQ_API_KEY is not set. AI features will use the non-AI fallback."
fi

echo "Starting HTML/CSS interface on http://127.0.0.1:8000"
python serve_frontend.py &
HTML_PID=$!

echo "Starting Streamlit interface on http://127.0.0.1:8501"
streamlit run app.py --server.port=8501 &
ST_PID=$!

trap "echo; echo 'Stopping...'; kill $HTML_PID $ST_PID 2>/dev/null" INT TERM
wait
