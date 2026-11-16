/**
 * Ada Intelligence Platform — Active Document Inventory Script
 */

(function () {
  const statusMessage = document.getElementById('organizer-status-message');
  const uploadInput = document.getElementById('organizer-upload-input');

  function showStatus(text, isError = false) {
    if (!statusMessage) return;
    statusMessage.style.display = 'block';
    statusMessage.textContent = text;
    statusMessage.style.color = isError ? 'var(--color-offline)' : 'var(--color-accent)';
  }

  // Document upload
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
})();
