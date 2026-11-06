/**
 * Ada Intelligence Platform — RAG Evaluation Benchmark Client Script
 */

(function () {
  let evaluationCases = [];

  const btnExecute = document.getElementById('btn-execute-eval');
  const btnReload = document.getElementById('btn-reload-eval');
  const pdfInput = document.getElementById('eval-pdf-input');
  const evalStatus = document.getElementById('eval-status-banner');
  const deepDiveInspector = document.getElementById('deep-dive-inspector');

  function showStatus(text, isError = false) {
    if (!evalStatus) return;
    evalStatus.style.display = 'block';
    evalStatus.textContent = text;
    evalStatus.style.color = isError ? 'var(--color-offline)' : 'var(--color-accent)';
  }

  // Execute benchmark
  if (btnExecute) {
    btnExecute.addEventListener('click', async () => {
      btnExecute.disabled = true;
      showStatus('Extracting grounded questions from uploaded PDF & evaluating RAG pipeline (retrieval, correctness, faithfulness, latency)...');
      try {
        const res = await fetch('/api/eval/', { method: 'POST' });
        const data = await res.json();
        showStatus('Benchmark completed successfully.');
        setTimeout(() => window.location.reload(), 800);
      } catch (err) {
        showStatus('Benchmark failed: ' + err.message, true);
        btnExecute.disabled = false;
      }
    });
  }

  // Reload results
  if (btnReload) {
    btnReload.addEventListener('click', () => {
      window.location.reload();
    });
  }

  // Upload PDF to benchmark dynamically
  if (pdfInput) {
    pdfInput.addEventListener('change', async (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      showStatus(`Ingesting ${file.name}, extracting questions, and executing benchmark...`);
      const formData = new FormData();
      formData.append('file', file);

      try {
        const res = await fetch('/api/eval/upload/', {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();
        showStatus('Dynamic benchmark completed successfully.');
        setTimeout(() => window.location.reload(), 1000);
      } catch (err) {
        showStatus('Upload and eval error: ' + err.message, true);
      }
    });
  }

  // Deep-dive inspector selection
  window.selectTestCase = function (caseDataJson) {
    let c = caseDataJson;
    if (typeof caseDataJson === 'string') {
      try {
        c = JSON.parse(decodeURIComponent(caseDataJson));
      } catch (e) {
        return;
      }
    }

    if (!c || !deepDiveInspector) return;

    deepDiveInspector.style.display = 'block';
    document.getElementById('inspect-qid').textContent = c.id || '';
    document.getElementById('inspect-query').textContent = c.query || '';
    document.getElementById('inspect-ref').textContent = c.reference_answer || 'N/A';
    document.getElementById('inspect-gen').textContent = c.generated_answer || c.answer || 'None';
    document.getElementById('inspect-citations').textContent = (c.citations || []).join(', ') || 'None';

    const timing = c.timing_ms || {};
    document.getElementById('inspect-timing').textContent = `Total ${(timing.total || 0).toFixed(1)}ms (Retrieval: ${(timing.retrieval || 0).toFixed(1)}ms, Generation: ${(timing.generation || 0).toFixed(1)}ms)`;

    // Scroll to inspector
    deepDiveInspector.scrollIntoView({ behavior: 'smooth' });
  };
})();
