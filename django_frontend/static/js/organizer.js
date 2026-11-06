/**
 * Ada Intelligence Platform — File Organizer Client Script
 */

(function () {
  let currentProposals = [];

  const folderInput = document.getElementById('target-folder-input');
  const btnBrowse = document.getElementById('btn-browse-folder');
  const btnAnalyze = document.getElementById('btn-analyze');
  const proposalsSection = document.getElementById('proposals-section');
  const proposalsTableBody = document.getElementById('proposals-table-body');
  const btnApplyMoves = document.getElementById('btn-apply-moves');
  const statusMessage = document.getElementById('organizer-status-message');
  const uploadInput = document.getElementById('organizer-upload-input');

  function showStatus(text, isError = false) {
    if (!statusMessage) return;
    statusMessage.style.display = 'block';
    statusMessage.textContent = text;
    statusMessage.style.color = isError ? 'var(--color-offline)' : 'var(--color-accent)';
  }

  function hideStatus() {
    if (statusMessage) statusMessage.style.display = 'none';
  }

  // Pick folder using system folder picker dialog
  if (btnBrowse) {
    btnBrowse.addEventListener('click', async () => {
      try {
        const res = await fetch('/api/folders/pick/', { method: 'POST' });
        const data = await res.json();
        if (data.picked && folderInput) {
          folderInput.value = data.picked;
          analyzeDirectory(data.picked);
        }
      } catch (err) {
        showStatus('Folder picker error: ' + err.message, true);
      }
    });
  }

  // Analyze directory
  async function analyzeDirectory(folder) {
    if (!folder) return;
    hideStatus();
    showStatus('Scanning and analyzing files across directory taxonomy...');

    if (btnAnalyze) btnAnalyze.disabled = true;

    try {
      const res = await fetch('/api/folders/refresh/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ folder: folder }),
      });
      const data = await res.json();

      currentProposals = data.suggestions || [];
      renderProposals(currentProposals);

      if (currentProposals.length === 0) {
        showStatus('Directory is clean or has no unorganized loose files.');
      } else {
        showStatus(`Found ${currentProposals.length} organization proposals.`);
      }
    } catch (err) {
      showStatus('Analysis failed: ' + err.message, true);
    } finally {
      if (btnAnalyze) btnAnalyze.disabled = false;
    }
  }

  if (btnAnalyze) {
    btnAnalyze.addEventListener('click', () => {
      const folder = folderInput ? folderInput.value.trim() : '';
      analyzeDirectory(folder);
    });
  }

  function renderProposals(proposals) {
    if (!proposalsTableBody || !proposalsSection) return;

    if (proposals.length === 0) {
      proposalsSection.style.display = 'none';
      return;
    }

    proposalsSection.style.display = 'block';
    proposalsTableBody.innerHTML = '';

    proposals.forEach((p, idx) => {
      const tr = document.createElement('tr');
      const filename = p.path ? p.path.split(/[\\/]/).pop() : 'file';

      tr.innerHTML = `
        <td style="width: 36px; text-align: center;">
          <input type="checkbox" class="proposal-checkbox" data-idx="${idx}" checked style="cursor: pointer;">
        </td>
        <td style="font-weight: 600; color: var(--text-bright);">${escapeHtml(filename)}</td>
        <td style="color: var(--color-accent); font-family: var(--font-mono); font-size: 11px;">${escapeHtml(p.suggested_destination || '')}</td>
        <td style="color: var(--text-muted);">${escapeHtml(p.reason || 'Classification')}</td>
      `;
      proposalsTableBody.appendChild(tr);
    });
  }

  // Apply approved moves
  if (btnApplyMoves) {
    btnApplyMoves.addEventListener('click', async () => {
      const checkboxes = document.querySelectorAll('.proposal-checkbox:checked');
      if (checkboxes.length === 0) {
        alert('Please select at least one move proposal to apply.');
        return;
      }

      const rowsToApply = [];
      checkboxes.forEach((cb) => {
        const idx = parseInt(cb.getAttribute('data-idx'), 10);
        if (currentProposals[idx]) {
          rowsToApply.push(currentProposals[idx]);
        }
      });

      if (!confirm(`Apply ${rowsToApply.length} approved relocation(s)?`)) return;

      btnApplyMoves.disabled = true;
      showStatus('Relocating approved files non-destructively...');

      try {
        const res = await fetch('/api/apply/', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ rows: rowsToApply }),
        });
        const data = await res.json();
        showStatus(`Successfully relocated ${data.applied || rowsToApply.length} file(s).`);

        // Refresh proposals
        const folder = folderInput ? folderInput.value.trim() : '';
        if (folder) analyzeDirectory(folder);
      } catch (err) {
        showStatus('Failed to apply moves: ' + err.message, true);
      } finally {
        btnApplyMoves.disabled = false;
      }
    });
  }

  // Document upload in Organizer panel
  if (uploadInput) {
    uploadInput.addEventListener('change', async (e) => {
      const files = e.target.files;
      if (!files || files.length === 0) return;

      showStatus('Uploading and indexing files...');
      const formData = new FormData();
      for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
      }

      try {
        const res = await fetch('/api/ingest/', { method: 'POST', body: formData });
        const data = await res.json();
        showStatus(`Indexed ${(data.ingested || []).length} document(s).`);
        setTimeout(() => window.location.reload(), 1200);
      } catch (err) {
        showStatus('Upload error: ' + err.message, true);
      }
    });
  }

  // Delete uploaded file helper
  window.deleteUploadedDoc = async function (filename) {
    if (!confirm(`Delete ${filename} from active index?`)) return;
    try {
      const res = await fetch('/api/ingest/' + encodeURIComponent(filename) + '/', {
        method: 'DELETE',
      });
      if (res.ok) {
        window.location.reload();
      }
    } catch (err) {
      alert('Delete failed: ' + err.message);
    }
  };

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }
})();
