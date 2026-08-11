FROM python:3.11-slim

# System libraries needed by PyMuPDF / Pillow / OpenCV (an EasyOCR dependency)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so Docker caches this layer between builds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 7860 is what Hugging Face Spaces expects.
# Render / Railway / Heroku inject their own $PORT, which overrides this.
ENV PORT=7860
ENV HOST=0.0.0.0
EXPOSE 7860

# Runs the HTML/CSS app. To deploy the Streamlit one instead, swap this for:
#   CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
CMD ["python", "serve_frontend.py"]
