/**
 * Ada Intelligence Platform — Chat Interface & Real-time SSE Streaming
 * Features:
 * - Real-time Server-Sent Events (SSE) streaming
 * - Adaptive multi-step tool execution feedback
 * - File attachments via paperclip picker and drag-and-drop
 * - Verified source citation pills
 * - Zero deliverable UI clutter
 */

(function () {
  const TOOL_LABELS = {
    search_documents: 'Searching documents',
    read_chunk: 'Reading passage',
    read_chunks: 'Reading multiple passages',
    list_folder: 'Listing folder contents',
    read_file: 'Reading file',
    grep_files: 'Searching within files',
    write_note: 'Saving analysis notes',
  };

  const ALLOWED_EXTENSIONS = [
    'pdf', 'xlsx', 'xls', 'csv', 'tsv', 'docx', 'doc', 'pptx', 'ppt',
    'txt', 'md', 'py', 'json', 'png', 'jpg', 'jpeg', 'html', 'xml', 'zip'
  ];
  const MAX_FILE_SIZE = 150 * 1024 * 1024; // 150 MB

  let activeController = null;
  let isSending = false;
  let stagedFiles = [];

  const mainContainer = document.getElementById('chat-container') || document.querySelector('.chat-container');
  const chatMessages = document.getElementById('chat-messages');
  const chatInput = document.getElementById('chat-input');
  const sendBtn = document.getElementById('btn-send');
  const attachBtn = document.getElementById('btn-attach');
  const fileInput = document.getElementById('chat-file-input');
  const attachmentPreview = document.getElementById('attachment-preview');
  const dropOverlay = document.getElementById('chat-drop-overlay');
  const ingestBanner = document.getElementById('ingest-banner');
  const activeSessionId = window.ADA_SESSION_ID || 'default';

  function scrollToBottom() {
    if (chatMessages) {
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
  }

  function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return (bytes / Math.pow(k, i)).toFixed(i > 1 ? 1 : 0) + ' ' + sizes[i];
  }

  // Minimal Markdown-to-HTML parser (fenced code, bold, inline code, lists)
  function renderMarkdown(text) {
    if (!text) return '';
    let out = text;

    // Fenced code blocks
    out = out.replace(/```([a-zA-Z0-9_]*)\n([\s\S]*?)```/g, function (_, lang, code) {
      return `<pre><code>${escapeHtml(code.trim())}</code></pre>`;
    });

    // Inline code
    out = out.replace(/`([^`]+)`/g, function (_, code) {
      return `<code>${escapeHtml(code)}</code>`;
    });

    // Bold text
    out = out.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Bullet lists
    out = out.replace(/(?:^|\n)[-*]\s+(.+)/g, '\n<li>$1</li>');
    out = out.replace(/(<li>.+<\/li>)+/g, '<ul>$&</ul>');

    // Paragraph line breaks
    out = out.replace(/\n{2,}/g, '<br><br>');
    out = out.replace(/(?<!<\/li>)\n(?!<li|<\/ul>)/g, '<br>');

    return out;
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // Staged Files Management
  function addStagedFiles(fileList) {
    if (!fileList || fileList.length === 0) return;

    const invalidExts = [];
    const oversized = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const ext = (file.name.split('.').pop() || '').toLowerCase();

      if (!ALLOWED_EXTENSIONS.includes(ext)) {
        invalidExts.push(file.name);
        continue;
      }

      if (file.size > MAX_FILE_SIZE) {
        oversized.push(`${file.name} (${formatBytes(file.size)})`);
        continue;
      }

      // Check duplicates
      const exists = stagedFiles.some(f => f.name === file.name && f.size === file.size);
      if (!exists) {
        stagedFiles.push(file);
      }
    }

    if (invalidExts.length > 0 || oversized.length > 0) {
      let msg = '';
      if (invalidExts.length > 0) {
        msg += `Unsupported file type: ${invalidExts.join(', ')}. `;
      }
      if (oversized.length > 0) {
        msg += `File exceeds 150MB limit: ${oversized.join(', ')}.`;
      }
      showBanner(msg, 'error');
    }

    renderAttachmentPreview();
  }

  function removeStagedFile(index) {
    if (index >= 0 && index < stagedFiles.length) {
      stagedFiles.splice(index, 1);
      renderAttachmentPreview();
    }
  }

  function renderAttachmentPreview() {
    if (!attachmentPreview) return;

    if (stagedFiles.length === 0) {
      attachmentPreview.style.display = 'none';
      attachmentPreview.innerHTML = '';
      return;
    }

    attachmentPreview.style.display = 'flex';
    attachmentPreview.innerHTML = '';

    stagedFiles.forEach((file, idx) => {
      const chip = document.createElement('div');
      chip.className = 'attachment-chip';

      const iconSpan = document.createElement('span');
      iconSpan.className = 'chip-icon';
      iconSpan.innerHTML = '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'chip-name';
      nameSpan.textContent = file.name;
      nameSpan.title = file.name;

      const sizeSpan = document.createElement('span');
      sizeSpan.className = 'chip-size';
      sizeSpan.textContent = formatBytes(file.size);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'chip-remove';
      removeBtn.title = 'Remove attachment';
      removeBtn.innerHTML = '✕';
      removeBtn.onclick = (e) => {
        e.stopPropagation();
        removeStagedFile(idx);
      };

      chip.appendChild(iconSpan);
      chip.appendChild(nameSpan);
      chip.appendChild(sizeSpan);
      chip.appendChild(removeBtn);
      attachmentPreview.appendChild(chip);
    });
  }

  function showBanner(msg, type = 'info') {
    if (!ingestBanner) return;
    ingestBanner.style.display = 'flex';
    const colorStyle = type === 'error' ? 'style="color: var(--color-offline);"' : '';
    ingestBanner.innerHTML = `<span ${colorStyle}>${escapeHtml(msg)}</span><button onclick="this.parentElement.style.display='none'" class="btn-icon-sm">✕</button>`;
  }

  // Create message element
  function appendUserMessage(text, attachedList) {
    const row = document.createElement('div');
    row.className = 'message-row user';

    const avatar = document.createElement('div');
    avatar.className = 'avatar user';
    avatar.textContent = 'U';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = renderMarkdown(text);

    if (attachedList && attachedList.length > 0) {
      const attachBox = document.createElement('div');
      attachBox.className = 'message-attachments';

      attachedList.forEach((file) => {
        const pill = document.createElement('span');
        pill.className = 'attached-pill';
        pill.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg> ${escapeHtml(file.name)} <span style="opacity: 0.7; font-size: 10px;">(${formatBytes(file.size)})</span>`;
        attachBox.appendChild(pill);
      });

      bubble.appendChild(attachBox);
    }

    row.appendChild(avatar);
    row.appendChild(bubble);

    chatMessages.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  // Send Chat message with SSE Streaming
  async function sendMessage() {
    const rawMessage = chatInput ? chatInput.value.trim() : '';
    const hasFiles = stagedFiles.length > 0;

    if ((!rawMessage && !hasFiles) || isSending) return;

    if (chatInput) chatInput.value = '';
    isSending = true;
    updateSendButtonState(true);

    // Stash files to upload
    const filesToUpload = [...stagedFiles];
    stagedFiles = [];
    renderAttachmentPreview();

    let displayMessage = rawMessage;
    if (!displayMessage && hasFiles) {
      displayMessage = `Please analyze the attached document${filesToUpload.length > 1 ? 's' : ''}: ${filesToUpload.map(f => f.name).join(', ')}`;
    }

    // Append User Message with attachment pills
    appendUserMessage(displayMessage, filesToUpload);

    // Create Assistant Row with Typing/Activity Container
    const row = document.createElement('div');
    row.className = 'message-row assistant';

    const avatar = document.createElement('div');
    avatar.className = 'avatar assistant';
    avatar.textContent = 'A';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Activity timeline for tools
    const activityTrail = document.createElement('div');
    activityTrail.className = 'activity-trail';

    // Live tokens container
    const liveContent = document.createElement('div');
    liveContent.className = 'live-content';
    liveContent.innerHTML = '<span class="typing-indicator"><span class="typing-dot"></span><span class="typing-dot" style="animation-delay: 150ms"></span><span class="typing-dot" style="animation-delay: 300ms"></span></span>';

    bubble.appendChild(activityTrail);
    bubble.appendChild(liveContent);
    row.appendChild(avatar);
    row.appendChild(bubble);
    chatMessages.appendChild(row);
    scrollToBottom();

    // If files were attached, upload & synchronously index them first
    let attachedFilenames = [];
    if (filesToUpload.length > 0) {
      showBanner(`Indexing ${filesToUpload.length} attached document(s)...`);
      const uploadItem = document.createElement('div');
      uploadItem.className = 'activity-item running';
      uploadItem.innerHTML = `<span class="activity-pulse-dot"></span><span>Ingesting & indexing attached documents...</span>`;
      activityTrail.appendChild(uploadItem);
      scrollToBottom();

      try {
        const formData = new FormData();
        for (const file of filesToUpload) {
          formData.append('files', file);
        }

        const ingestRes = await fetch('/api/ingest/', {
          method: 'POST',
          body: formData,
        });

        if (!ingestRes.ok) {
          throw new Error(`Ingest HTTP ${ingestRes.status}`);
        }

        const ingestData = await ingestRes.json();
        attachedFilenames = filesToUpload.map(f => f.name);

        uploadItem.classList.remove('running');
        uploadItem.innerHTML = `<span>Indexed ${filesToUpload.length} attachment(s)</span>`;
        if (ingestBanner) ingestBanner.style.display = 'none';
      } catch (uploadErr) {
        uploadItem.classList.remove('running');
        uploadItem.innerHTML = `<span style="color: var(--color-offline);">Attachment upload failed: ${escapeHtml(uploadErr.message)}</span>`;
        showBanner(`Upload failed: ${uploadErr.message}`, 'error');
        // Continue with chat even if attachment failed
      }
    }

    activeController = new AbortController();
    let accumulatedText = '';
    let citations = [];

    try {
      const response = await fetch('/api/chat/stream/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: activeSessionId,
          message: displayMessage,
          attachments: attachedFilenames,
        }),
        signal: activeController.signal,
      });

      if (!response.ok) {
        throw new Error('Server returned ' + response.status);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split('\n\n');
        buffer = frames.pop();

        for (const frame of frames) {
          const line = frame.split('\n').find((l) => l.startsWith('data: '));
          if (!line) continue;

          try {
            const ev = JSON.parse(line.slice(6));
            handleAgentEvent(ev, activityTrail, liveContent);

            if (ev.t === 'token' && ev.d) {
              accumulatedText += ev.d;
              liveContent.innerHTML = renderMarkdown(accumulatedText);
              scrollToBottom();
            } else if (ev.t === 'done') {
              accumulatedText = ev.answer || accumulatedText;
              citations = ev.citations || [];
              liveContent.innerHTML = renderMarkdown(accumulatedText);
              renderCitations(bubble, citations);
              scrollToBottom();
            } else if (ev.t === 'error') {
              liveContent.innerHTML = `<span style="color: var(--color-offline);">Error: ${escapeHtml(ev.message)}</span>`;
            }
          } catch (jsonErr) {
            // Ignore frame parse error
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        liveContent.innerHTML = `<span style="color: var(--color-offline);">Execution error: ${escapeHtml(err.message)}</span>`;
      }
    } finally {
      isSending = false;
      activeController = null;
      updateSendButtonState(false);
      scrollToBottom();
    }
  }

  function handleAgentEvent(ev, activityTrail, liveContent) {
    if (ev.t === 'tool_start') {
      const label = TOOL_LABELS[ev.name] || ev.name || 'Reasoning';
      const item = document.createElement('div');
      item.className = 'activity-item running';
      item.id = 'act-' + (ev.name || 'tool');
      item.innerHTML = `<span class="activity-pulse-dot"></span><span>${escapeHtml(label)}...</span>`;
      activityTrail.appendChild(item);
      scrollToBottom();
    } else if (ev.t === 'tool_end') {
      const items = activityTrail.querySelectorAll('.activity-item.running');
      if (items.length > 0) {
        const last = items[items.length - 1];
        last.classList.remove('running');
        const msText = ev.ms ? ` · ${(ev.ms / 1000).toFixed(1)}s` : '';
        last.innerHTML = `<span>${last.textContent.replace('...', '')}${msText}</span>`;
      }
    } else if (ev.t === 'verify') {
      const item = document.createElement('div');
      item.className = 'activity-item';
      item.innerHTML = '<span>Verifying claims against document sources</span>';
      activityTrail.appendChild(item);
    }
  }

  function renderCitations(bubble, citations) {
    if (citations && citations.length > 0) {
      const citBox = document.createElement('div');
      citBox.className = 'citation-list';
      citBox.innerHTML = '<span style="font-size: 11px; font-weight: 700; color: var(--text-subtle); text-transform: uppercase;">Verified Sources:</span>';

      citations.forEach((c) => {
        const pill = document.createElement('span');
        pill.className = 'citation-pill';
        pill.textContent = c;
        citBox.appendChild(pill);
      });
      bubble.appendChild(citBox);
    }
  }

  function stopStreaming() {
    if (activeController) {
      activeController.abort();
    }
  }

  function updateSendButtonState(running) {
    if (!sendBtn) return;
    if (running) {
      sendBtn.classList.add('stop');
      sendBtn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><rect x="5" y="5" width="14" height="14" rx="2"/></svg>';
      sendBtn.onclick = stopStreaming;
    } else {
      sendBtn.classList.remove('stop');
      sendBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5M5 12l7-7 7 7"/></svg>';
      sendBtn.onclick = sendMessage;
    }
  }

  // Setup Drag and Drop Listeners
  if (mainContainer && dropOverlay) {
    let dragCounter = 0;

    mainContainer.addEventListener('dragenter', (e) => {
      e.preventDefault();
      dragCounter++;
      dropOverlay.style.display = 'flex';
    });

    mainContainer.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'copy';
      dropOverlay.style.display = 'flex';
    });

    mainContainer.addEventListener('dragleave', (e) => {
      e.preventDefault();
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        dropOverlay.style.display = 'none';
      }
    });

    mainContainer.addEventListener('drop', (e) => {
      e.preventDefault();
      dragCounter = 0;
      dropOverlay.style.display = 'none';
      if (e.dataTransfer && e.dataTransfer.files) {
        addStagedFiles(e.dataTransfer.files);
      }
    });
  }

  // Event Listeners
  if (attachBtn && fileInput) {
    attachBtn.addEventListener('click', () => {
      fileInput.click();
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files) {
        addStagedFiles(e.target.files);
        fileInput.value = '';
      }
    });
  }

  if (sendBtn) {
    sendBtn.addEventListener('click', sendMessage);
  }

  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // Auto scroll on load
  document.addEventListener('DOMContentLoaded', () => {
    scrollToBottom();
  });
})();
