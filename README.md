---
title: AI Document Summarizer
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AI Document Summarizer & Q&A System

An AI-powered web application for intelligent document analysis. Upload a
document, get a summary and key points, and ask questions answered from the
document's own content with source citations.

There are **two interfaces**, both using the same engine:

| Interface | File | Run with | Default URL |
| --- | --- | --- | --- |
| Custom HTML/CSS (main) | `serve_frontend.py` | `python serve_frontend.py` | http://127.0.0.1:8000 |
| Streamlit (alternative) | `app.py` | `streamlit run app.py` | http://127.0.0.1:8501 |

You can run both at the same time — they use different ports.

## Features

- **Multi-format support**: PDF, DOCX, TXT, MD, JSON
- **Scanned PDF OCR**: automatic detection, handled by EasyOCR (no Tesseract needed)
- **Smart summaries**: multiple prompt variants
- **Key point extraction** with evidence tagging
- **Source-aware Q&A**: FAISS semantic search with page/paragraph/section citations
- **Graceful fallbacks**: works without an API key or the heavy ML libraries
- **Export**: TXT, DOCX, PDF

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

**Light install** (skips PyTorch, ~3 GB smaller). OCR and semantic search turn
off but nothing crashes:

```bash
pip install pypdf python-docx fpdf langchain langchain-core langchain-groq streamlit
```

### 2. Set your Groq API key

The key is read from the environment. Never hardcode it.

```bash
# Mac / Linux
export GROQ_API_KEY=your_key_here

# Windows (Command Prompt)
set GROQ_API_KEY=your_key_here

# Windows (PowerShell)
$env:GROQ_API_KEY="your_key_here"
```

Without a key the app still runs, but summaries and answers use a non-AI
fallback. The server prints a clear message telling you which mode it is in.

### 3. Run

**HTML/CSS interface:**

```bash
python serve_frontend.py
```

Open http://127.0.0.1:8000

**Streamlit interface:**

```bash
streamlit run app.py
```

Open http://127.0.0.1:8501

**Both at once** — use the helper script:

```bash
./run_both.sh      # Mac / Linux
run_both.cmd       # Windows
```

## Deployment

The included `Dockerfile` deploys the HTML/CSS interface and works on Hugging
Face Spaces, Render, Railway, and Fly.io.

- **Hugging Face Spaces**: keep the frontmatter above (`sdk: docker`,
  `app_port: 7860`). Add `GROQ_API_KEY` under Settings → Variables and secrets.
- **Render / Railway / Fly**: delete the frontmatter block above. Choose Docker
  as the environment. Add `GROQ_API_KEY` in the environment panel.

To deploy the Streamlit interface instead, change the last line of the
`Dockerfile` to the commented-out `streamlit run` command.

A host gives you **one port**, so only one interface can be live per
deployment. To have both online, create two separate deployments from the same
repository.

### Deployment size warning

`easyocr` and `sentence-transformers` both install PyTorch (~2-3 GB) and need
significant RAM. This will not fit on the free tier of Render or Railway.
Hugging Face's free CPU tier is large enough. For small tiers, remove
`easyocr`, `sentence-transformers`, and `faiss-cpu` from `requirements.txt`.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | *(unset)* | Enables AI features. Without it, fallbacks are used. |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | Which Groq model to call. |
| `PORT` | `8000` | Port for `serve_frontend.py`. |
| `HOST` | `0.0.0.0` | Bind address. |

## Architecture

- **Backend**: Python HTTP server (`serve_frontend.py`), no framework
- **Frontend**: HTML5 / CSS3 / vanilla JavaScript, no build step
- **AI**: LangChain + Groq
- **Retrieval**: sentence-transformers embeddings in a FAISS index
- **Extraction**: pypdf, PyMuPDF + EasyOCR (scanned), python-docx

## API Endpoints

- `GET /api/health` — returns `{"status": "ok"}`
- `POST /api/analyze` — multipart upload of one or more documents
- `GET /api/analyze-status?job=<id>` — progress for long-running uploads
- `POST /api/ask` — JSON with `question`, `context`, `chunks`, `chunk_records`, `qa_variant`
- `POST /api/export` — JSON with `summary`, `key_points`, `format` (`txt`/`docx`/`pdf`)

## Project Structure

```
AI_Document_summarizer/
├── serve_frontend.py   # HTTP backend + static file server (main app)
├── app.py              # Streamlit interface (alternative)
├── index.html          # Web interface
├── styles.css          # Styling
├── script.js           # Frontend logic
├── utils.py            # Document extraction and chunking
├── rag.py              # FAISS embeddings and retrieval
├── prompt_engineering.py  # Prompt variants and evaluation
├── document_evaluation.py # OCR accuracy comparison
├── Dockerfile          # Deployment
├── requirements.txt
└── tests/
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests for FAISS retrieval are skipped automatically when `faiss-cpu` and
`sentence-transformers` are not installed.

## Troubleshooting

**Summaries look like copied sentences from the document.**
The AI is off. Check the server startup output — it prints `[LLM] DISABLED`
with the reason. Usually `GROQ_API_KEY` is not set.

**Q&A says "Not found in the document" for obvious questions.**
Check that `retrieval_method` in the response is not `keyword fallback`. If it
is, install `faiss-cpu` and `sentence-transformers`.

**Deployed site shows no styling.**
The host is running the wrong app. Confirm the frontmatter says `sdk: docker`,
not `sdk: streamlit`.

## Author

Farzeen Fatima — BS Artificial Intelligence
