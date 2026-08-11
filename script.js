const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const selectedFiles = document.getElementById('selectedFiles');
const uploadBtn = document.getElementById('uploadBtn');
const exportSummaryBtn = document.getElementById('exportSummaryBtn');
const exportSummaryBtns = document.querySelectorAll('.export-summary-btn');
const summary = document.getElementById('summary');
const keyPoints = document.getElementById('keyPoints');
const statusMessage = document.getElementById('statusMessage');
const progressContainer = document.getElementById('progressContainer');
const uploadProgress = document.getElementById('uploadProgress');
const analysisProgress = document.getElementById('analysisProgress');
const progressStatus = document.getElementById('progressStatus');
const taskBadge = document.getElementById('taskBadge');
const summaryVariant = document.getElementById('summaryVariant');
const keyPointsVariant = document.getElementById('keyPointsVariant');
const qaVariant = document.getElementById('qaVariant');
const qaPromptVariant = document.getElementById('qaPromptVariant');
const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('question');
const conversationHistory = document.getElementById('conversationHistory');
const typingIndicator = document.getElementById('typingIndicator');
const copySummaryBtn = document.getElementById('copySummaryBtn');
const copyAnswerBtn = document.getElementById('copyAnswerBtn');
const exportFormat = document.getElementById('exportFormat');
const sourceCitations = document.getElementById('sourceCitations');
const documentPreview = document.getElementById('documentPreview');
const indexedChunks = document.getElementById('indexedChunks');
const retrievalContext = document.getElementById('retrievalContext');
const retrievedChunksList = document.getElementById('retrievedChunksList');
const ragInfo = document.getElementById('ragInfo');
const retrievalMethodEl = document.getElementById('retrievalMethod');
const ocrPagesValue = document.getElementById('ocrPages');
const STATUS_READY = 'Ready to upload';
const STATUS_ANALYZING = 'Analyzing document';
const STATUS_COMPLETED = 'Analysis complete';
const STATUS_ERROR = 'Problem encountered during analysis';
let uploadedFiles = [];
let analysisData = null;

const apiBase = window.location.origin;

const renderFileList = (files) => {
  selectedFiles.innerHTML = '';
  if (!files?.length) {
    selectedFiles.classList.add('placeholder');
    selectedFiles.innerHTML = '<p>No files selected yet.</p>';
    return;
  }

  selectedFiles.classList.remove('placeholder');
  const header = document.createElement('div');
  header.className = 'file-list-header';
  header.innerHTML = `<div><strong>${files.length}</strong> file${files.length === 1 ? '' : 's'} selected</div>`;
  selectedFiles.appendChild(header);

  files.forEach((file) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `<div><strong>${file.name}</strong><div><span>${(file.size / 1024).toFixed(1)} KB</span></div></div>`;
    selectedFiles.appendChild(item);
  });
};

const setStatus = (text, isError = false) => {
  statusMessage.textContent = text;
  statusMessage.style.color = isError ? '#ff8f8f' : 'var(--muted)';
};

const setProgress = (upload = 0, analysis = 0, label = STATUS_READY) => {
  progressContainer.classList.remove('hidden');
  uploadProgress.style.width = `${upload}%`;
  analysisProgress.style.width = `${analysis}%`;
  progressStatus.textContent = label;
  taskBadge.textContent = label === STATUS_COMPLETED ? 'Done' : 'In progress';
};

const clearAnalysis = () => {
  summary.innerHTML = 'Your summary will appear here after processing.';
  keyPoints.innerHTML = '<li>Upload a file to generate a summary and key points.</li>';
  conversationHistory.innerHTML = '<div class="chat-placeholder"><p>Ask a question about the uploaded document to get a focused answer.</p></div>';
  sourceCitations.innerHTML = 'Sources for each answer will appear here.';
  documentPreview.textContent = 'Document preview will appear here after upload.';
  indexedChunks.innerHTML = 'Indexed chunks will appear here after analysis.';
  retrievalMethodEl.textContent = 'No retrieval yet';
  ragInfo.style.display = 'none';
  retrievalContext.style.display = 'none';
  qaPromptVariant.textContent = qaVariant.value;
  if (ocrPagesValue) {
    ocrPagesValue.textContent = 'None';
  }
  updateMetaCards({
    extraction_method: 'Text layer',
    chunk_count: '0',
    document_length: '0 chars',
    qa_variant: qaVariant.value,
    ocr_pages: 'None',
  });
};

const updateMetaCards = (payload) => {
  const extractionMethod = document.getElementById('extractionMethodValue');
  const chunkCount = document.getElementById('chunkCountValue');
  const documentLength = document.getElementById('documentLengthValue');
  const ocrPages = document.getElementById('ocrPages');
  if (extractionMethod) extractionMethod.textContent = payload.extraction_method || 'Text layer';
  if (chunkCount) chunkCount.textContent = payload.chunk_count || '0';
  const documentCount = document.getElementById('documentCountValue');
  if (documentCount) documentCount.textContent = payload.document_count || '0';
  if (documentLength) documentLength.textContent = payload.document_length || '0 chars';
  if (ocrPages) ocrPages.textContent = payload.ocr_pages || 'None';
  qaPromptVariant.textContent = qaVariant.value;
};

const createMessage = (text, type = 'answer', error = false) => {
  const wrapper = document.createElement('div');
  wrapper.className = 'message-pair';
  const message = document.createElement('div');
  message.className = `message ${type}-message${error ? ' error' : ''}`;
  message.textContent = text;
  wrapper.appendChild(message);
  return wrapper;
};

const addChatMessage = (text, type) => {
  if (conversationHistory.querySelector('.chat-placeholder')) {
    conversationHistory.innerHTML = '';
  }
  conversationHistory.appendChild(createMessage(text, type));
  conversationHistory.scrollTop = conversationHistory.scrollHeight;
};

const handleFileSelection = (files) => {
  if (!files?.length) return;
  const isNewSelection =
    !uploadedFiles ||
    uploadedFiles.length !== files.length ||
    files.some((file, idx) => {
      const existing = uploadedFiles[idx];
      return !existing || existing.name !== file.name || existing.size !== file.size;
    });

  uploadedFiles = files;
  renderFileList(files);
  if (isNewSelection) {
    analysisData = null;
    clearAnalysis();
  }
  setStatus(`Ready to analyze ${files.length} file${files.length === 1 ? '' : 's'}.`);
};

const handleAnalysisError = (error) => {
  const message = error?.message || String(error) || 'Analysis failed.';
  const friendly = message.includes('status 0')
    ? 'Backend unreachable. Ensure the server is running at http://127.0.0.1:8000'
    : message;
  setStatus(friendly, true);
  setProgress(100, 0, STATUS_ERROR);
  console.error('[Analysis Error]', error);
};

const uploadDocument = async () => {
  if (!uploadedFiles?.length) {
    setStatus('Select or drop one or more documents first.', true);
    return;
  }

  const formData = new FormData();
  uploadedFiles.forEach((file) => formData.append('file', file));
  formData.append('summary_prompt_variant', summaryVariant.value);
  formData.append('key_points_prompt_variant', keyPointsVariant.value);

  try {
    setStatus('Preparing upload...');
    setProgress(2, 4, 'Uploading');
    uploadBtn.disabled = true;

    // Use XMLHttpRequest to get upload progress events for a faster, smoother UX
    const xhr = new XMLHttpRequest();
    let analysisInterval = null;
    xhr.open('POST', `${apiBase}/api/analyze`, true);

    xhr.upload.onprogress = function (event) {
      if (event.lengthComputable) {
        const percent = Math.floor((event.loaded / event.total) * 100);
        uploadProgress.style.width = `${percent}%`;
        setStatus(`Uploading... ${percent}%`);
      }
    };

    xhr.onreadystatechange = function () {
      if (xhr.readyState === 2) {
        // Headers received
        setStatus('Upload sent, awaiting analysis...');
        setProgress(100, 6, 'Analyzing');
        // Start animated analysis progress to indicate work in background
        let anim = 6;
        analysisInterval = setInterval(() => {
          anim = Math.min(95, anim + Math.random() * 6);
          analysisProgress.style.width = `${Math.floor(anim)}%`;
        }, 600);
      }

      if (xhr.readyState === 4) {
        if (analysisInterval) clearInterval(analysisInterval);
        uploadBtn.disabled = false;
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const startResp = JSON.parse(xhr.responseText || '{}');
            // If server returned a job id, poll status endpoint for real progress
            if (startResp && startResp.job) {
              const jobId = startResp.job;
              setStatus('Analysis started, tracking progress...');
              // Poll status endpoint
              let pollRetries = 0;
              const poll = setInterval(async () => {
                try {
                  const statusResp = await fetch(`${apiBase}/api/analyze-status?job=${encodeURIComponent(jobId)}`);
                  if (!statusResp.ok) {
                    throw new Error(`Status request failed: ${statusResp.status}`);
                  }
                  const status = await statusResp.json();
                  const pct = Math.min(100, Math.max(0, status.progress || 0));
                  analysisProgress.style.width = `${pct}%`;
                  setProgress(100, pct, `Analyzing ${pct}%`);
                  if (status.status === 'completed' && status.result) {
                    clearInterval(poll);
                    const result = status.result;
                    summary.innerHTML = result.summary || 'No summary returned from the server.';
                    keyPoints.innerHTML = '';
                    if (Array.isArray(result.key_points)) {
                      result.key_points.forEach((point) => {
                        const li = document.createElement('li');
                        li.textContent = point;
                        keyPoints.appendChild(li);
                      });
                    }
                    const payload = {
    extraction_method: result.extraction_method || 'Text layer',
    chunk_count: result.chunk_count || 0,
    document_length: `${result.text_length || 0} chars`,
    document_count: Array.isArray(result.file_names) ? result.file_names.length : 0,
};
                    updateMetaCards(payload);
                    analysisData = result;
                    analysisData.chunk_records = result.chunk_records || [];
                    analysisData.retrieved_chunks = result.retrieved_chunks || [];
                    renderDocumentPreview(result);
                    renderIndexedChunks(result);
                    setProgress(100, 100, STATUS_COMPLETED);
                    setStatus('Document analysis complete. Ask a question or export the summary.');
                  } else if (status.status === 'error') {
                    clearInterval(poll);
                    handleAnalysisError(new Error(status.error || 'Analysis failed'));
                  } else {
                    pollRetries += 1;
                    if (pollRetries > 90) {
                      clearInterval(poll);
                      handleAnalysisError(new Error('Analysis is taking longer than expected. Please try again.'));
                    }
                  }
                } catch (err) {
                  clearInterval(poll);
                  handleAnalysisError(err);
                }
              }, 800);
            } else {
              // Backwards-compatible: server returned full result synchronously
              const result = startResp;
              summary.innerHTML = result.summary || 'No summary returned from the server.';
              keyPoints.innerHTML = '';
              if (Array.isArray(result.key_points)) {
                result.key_points.forEach((point) => {
                  const li = document.createElement('li');
                  li.textContent = point;
                  keyPoints.appendChild(li);
                });
              }
              const payload = {
    extraction_method: result.extraction_method || 'Text layer',
    chunk_count: result.chunk_count || 0,
    document_length: `${result.text_length || 0} chars`,
    document_count: Array.isArray(result.file_names) ? result.file_names.length : 0,
};
              updateMetaCards(payload);
              analysisData = result;
              analysisData.chunk_records = result.chunk_records || [];
              analysisData.retrieved_chunks = result.retrieved_chunks || [];
              renderDocumentPreview(result);
              renderIndexedChunks(result);
              setProgress(100, 100, STATUS_COMPLETED);
              setStatus('Document analysis complete. Ask a question or export the summary.');
            }
          } catch (err) {
            handleAnalysisError(err);
          }
        } else {
          const statusCode = xhr.status || '0';
          const errorText = xhr.responseText ? xhr.responseText : 'No response body';
          let message = errorText;
          try {
            const parsed = JSON.parse(errorText);
            if (parsed.error) {
              message = parsed.error;
            }
          } catch (e) {
            // ignore parse errors
          }
          handleAnalysisError(new Error(`Upload request failed: ${statusCode}. ${message}`));
        }
      }
    };

    xhr.onerror = function (ev) {
      if (analysisInterval) clearInterval(analysisInterval);
      uploadBtn.disabled = false;
      handleAnalysisError(new Error('Network error during upload'));
    };

    xhr.send(formData);
  } catch (error) {
    handleAnalysisError(error);
  }
};

// Improve perceived analysis speed by rapidly updating the analysis progress
const startAnalysisAnimation = () => {
  let pct = 4;
  analysisProgress.style.width = `${pct}%`;
  return setInterval(() => {
    pct = Math.min(96, pct + Math.random() * 5);
    analysisProgress.style.width = `${Math.floor(pct)}%`;
  }, 700);
};

const stopAnalysisAnimation = (intervalId) => {
  if (intervalId) clearInterval(intervalId);
};

const renderDocumentPreview = (result) => {
  if (!documentPreview) return;
  const preview = result.preview || result.text || '';
  documentPreview.textContent = preview || 'No preview available.';
  documentPreview.parentElement.style.display = preview ? 'block' : 'none';
};

const renderIndexedChunks = (result) => {
  if (!indexedChunks) return;
  const records = result.chunk_records || [];
  if (!records.length) {
    indexedChunks.innerHTML = '<p>No indexed chunks available.</p>';
    return;
  }
  const container = document.createElement('div');
  container.className = 'chunks-list';
  records.forEach((r, idx) => {
    const chunkText = (r.text || r.chunk || '').trim();
    const charCount = chunkText.length;
    const wordCount = chunkText.split(/\s+/).filter(Boolean).length;
    const section = r.section || 'Document';
    const fileName = r.file_name || 'Document';
    const el = document.createElement('div');
    el.className = 'chunk-item';
    el.innerHTML = `
      <div class="chunk-meta"><strong>Chunk ${idx + 1}</strong> — <span>Source: ${escapeHtml(fileName)}</span> • <span>Section: ${escapeHtml(section)}</span> • <span>Page: ${r.page_number || 'N/A'}</span> • <span>Para: ${r.paragraph_number || 'N/A'}</span> • <span>Chars: ${charCount}</span> • <span>Words: ${wordCount}</span></div>
      <div class="chunk-text">${escapeHtml(chunkText)}</div>
    `;
    container.appendChild(el);
  });
  indexedChunks.innerHTML = '';
  indexedChunks.appendChild(container);
};

const escapeHtml = (unsafe) => {
  return String(unsafe)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
};

const askQuestion = async () => {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus('Type a question before asking.', true);
    return;
  }
  if (!analysisData) {
    setStatus('Upload and analyze a document before asking questions.', true);
    return;
  }

  addChatMessage(question, 'question');
  questionInput.value = '';
  typingIndicator.classList.remove('hidden');

  try {
    const response = await fetch(`${apiBase}/api/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
        qa_variant: qaVariant.value,
        variant: qaVariant.value,
        context: analysisData.text || analysisData.preview || '',
        text: analysisData.text || analysisData.preview || '',
        chunk_records: analysisData.chunk_records || analysisData.chunks || [],
        chunks: analysisData.chunk_records || analysisData.chunks || [],
      }),
    });

    typingIndicator.classList.add('hidden');

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || 'QA request failed');
    }

    const result = await response.json();
    addChatMessage(result.answer || 'No answer returned.', 'answer');
    // Display RAG info and retrieval context inside the QA section
    const method = result.retrieval_method || result.retrievalMethod || 'N/A';
    retrievalMethodEl.textContent = method;
    ragInfo.style.display = 'block';
    retrievalContext.style.display = 'block';

    const retrieved = result.retrieved_chunks || result.retrievedChunks || [];
    const citations = result.citations || [];
    // Populate retrieved chunks list
    if (retrievedChunksList) {
      retrievedChunksList.innerHTML = '';
      const list = document.createElement('div');
      list.className = 'retrieved-list';
      (citations.length ? citations : retrieved).forEach((item, i) => {
        const score = (item.score || item.similarity || 0) * 100;
        const excerpt = item.excerpt || item.chunk || (item.text || '').slice(0, 350);
        const fileName = item.file_name || item.source || 'Unknown';
        const page = item.page_number || item.page || 'N/A';
        const paragraph = item.paragraph_number || item.paragraph || 'N/A';
        const section = item.section || 'Document';
        const rank = i + 1;
        const node = document.createElement('div');
        node.className = 'retrieved-item';
        node.innerHTML = `
          <div class="retrieved-meta"><strong>#${rank}</strong> — <span>${fileName}</span> • <span>Page ${page}</span> • <span>Para ${paragraph}</span> • <span>Section: ${section}</span> • <span>Score: ${isFinite(score) ? score.toFixed(1) + '%' : 'N/A'}</span></div>
          <div class="retrieved-excerpt">${escapeHtml(excerpt)}</div>
          <div class="retrieved-explain">Retrieved because it matched the question closely and ranks ${rank}.</div>
        `;
        list.appendChild(node);
      });
      retrievedChunksList.appendChild(list);
    }

    // Render detailed source citations panel
    sourceCitations.innerHTML = '';
    if (citations.length) {
      const citeContainer = document.createElement('div');
      citations.forEach((c, idx) => {
        const sscore = (c.score || 0) * 100;
        const chunkId = c.chunk_id || `${idx + 1}`;
        const p = document.createElement('div');
        p.className = 'citation-item';
        p.innerHTML = `
          <strong>Answer source ${idx + 1}</strong>
          <div>Page: ${c.page_number || 'N/A'} • Paragraph: ${c.paragraph_number || 'N/A'} • Section: ${c.section || 'Document'}</div>
          <div>Chunk ID: ${chunkId} • Source: ${c.file_name || 'Unknown'} • Similarity: ${isFinite(sscore) ? sscore.toFixed(1) + '%' : 'N/A'}</div>
          <div class="citation-excerpt">${escapeHtml(c.excerpt || '')}</div>
          <div class="citation-explain">Retrieved because it was a top-ranked relevant chunk.</div>
        `;
        citeContainer.appendChild(p);
      });
      sourceCitations.appendChild(citeContainer);
    } else if (retrieved.length) {
      sourceCitations.textContent = 'Retrieved context is available but no formal citations were returned.';
    } else {
      sourceCitations.textContent = 'No sources available for this answer.';
    }

    setStatus('Question answered successfully.');
  } catch (error) {
    typingIndicator.classList.add('hidden');
    addChatMessage('Unable to fetch an answer at this time.', 'answer', true);
    handleAnalysisError(error);
  }
};

const exportSummary = async () => {
  if (!analysisData) {
    setStatus('Perform an analysis before exporting the summary.', true);
    return;
  }

  try {
    const body = {
      summary: analysisData.summary || '',
      key_points: analysisData.key_points || [],
      text: analysisData.text || analysisData.preview || '',
      retrieved_chunks: analysisData.retrieved_chunks || [],
      file_name: (analysisData.file_names && analysisData.file_names[0]) || analysisData.file_name || 'document-summary',
      format: exportFormat.value || 'txt',
    };

    console.log('Export summary request body:', body);
    const response = await fetch(`${apiBase}/api/export`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const errorText = await response.text();
      let message = errorText;
      try {
        const parsed = JSON.parse(errorText);
        if (parsed.error) {
          message = parsed.error;
        }
      } catch (e) {
        // keep raw text if JSON parse fails
      }
      console.error('Export failed:', response.status, message);
      throw new Error(message || 'Export request failed');
    }

    const contentType = response.headers.get('Content-Type') || '';
    if (contentType.includes('application/json')) {
      const json = await response.json();
      const message = json.error || json.message || 'Export request returned JSON instead of file content.';
      console.error('Export returned JSON:', json);
      throw new Error(message);
    }

    const blob = await response.blob();
    const filename = `document-summary.${exportFormat.value}`;
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    console.log('Export successful:', filename, response.headers.get('Content-Disposition'));
    setStatus(`Export ready: ${filename}`);
  } catch (error) {
    console.error('Export Summary error:', error);
    handleAnalysisError(error);
  }
};

const copyText = async (text, label) => {
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`${label} copied to clipboard.`);
  } catch (error) {
    setStatus(`Unable to copy ${label}.`, true);
  }
};

const initialize = () => {
  renderFileList([]);
  clearAnalysis();
  qaPromptVariant.textContent = qaVariant.value;

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (event) => {
    event.preventDefault();
    dropZone.classList.add('drag-over');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropZone.classList.remove('drag-over');
    const files = Array.from(event.dataTransfer.files || []);
    if (files.length) {
      handleFileSelection(files);
    }
  });

  fileInput.addEventListener('change', (event) => {
    const files = Array.from(event.target.files || []);
    handleFileSelection(files);
  });

  uploadBtn.addEventListener('click', uploadDocument);
  askBtn.addEventListener('click', askQuestion);
  exportSummaryBtns.forEach((button) => button.addEventListener('click', exportSummary));
  copySummaryBtn.addEventListener('click', () => copyText(summary.textContent, 'summary'));
  copyAnswerBtn.addEventListener('click', () => {
    const lastAnswer = Array.from(conversationHistory.querySelectorAll('.answer-message')).pop();
    if (lastAnswer) {
      copyText(lastAnswer.textContent, 'answer');
    } else {
      setStatus('No answer available to copy.', true);
    }
  });

  summaryVariant.addEventListener('change', () => setStatus('Summary style updated.'));
  keyPointsVariant.addEventListener('change', () => setStatus('Key points style updated.'));
  qaVariant.addEventListener('change', () => {
    qaPromptVariant.textContent = qaVariant.value;
    setStatus('QA prompt updated.');
  });
};

initialize();

