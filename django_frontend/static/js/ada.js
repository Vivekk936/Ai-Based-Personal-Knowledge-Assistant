/**
 * Ada Intelligence Platform — Global Client Script
 * Handles backend status polling, loading screen state, and session operations.
 */

window.Ada = window.Ada || {};

(function () {
  const POLLING_INTERVAL_MS = 4000;

  // Global state
  Ada.state = {
    connected: false,
    health: null,
    everConnected: false,
  };

  // Poll backend health status
  async function pollHealth() {
    try {
      const res = await fetch('/api/health/');
      if (!res.ok) throw new Error('Health check failed: ' + res.status);
      const data = await res.json();
      
      Ada.state.health = data;
      Ada.state.connected = data.status === 'ok' || data.model_backend_reachable;
      Ada.state.everConnected = true;

      updateStatusUI(Ada.state.connected ? 'connected' : 'degraded', data);
      hideLoadingOverlay();
    } catch (err) {
      Ada.state.connected = false;
      const statusType = Ada.state.everConnected ? 'reconnecting' : 'connecting';
      updateStatusUI(statusType, null);
    } finally {
      setTimeout(pollHealth, POLLING_INTERVAL_MS);
    }
  }

  function updateStatusUI(statusType, health) {
    const dot = document.getElementById('status-dot');
    const label = document.getElementById('status-label');
    const model = document.getElementById('status-model');

    if (!dot || !label) return;

    dot.className = 'status-dot';

    if (statusType === 'connected') {
      dot.classList.add('online');
      label.textContent = 'Connected';
      if (model && health) {
        const chatModelName = health.chat_model ? health.chat_model.split('/').pop() : 'NVIDIA NIM';
        model.textContent = '· ' + chatModelName;
      }
    } else if (statusType === 'degraded') {
      dot.classList.add('degraded');
      label.textContent = 'Model Degraded / Connecting';
    } else if (statusType === 'reconnecting') {
      dot.classList.add('offline', 'pulse');
      label.textContent = 'Reconnecting...';
      if (model) model.textContent = '';
    } else {
      dot.classList.add('offline', 'pulse');
      label.textContent = 'Starting Ada...';
      if (model) model.textContent = '';
    }
  }

  function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay && !overlay.classList.contains('hidden')) {
      overlay.style.opacity = '0';
      overlay.style.transition = 'opacity 0.25s ease';
      setTimeout(() => overlay.remove(), 260);
    }
  }

  // Session Management Helpers
  Ada.createSession = async function () {
    try {
      const res = await fetch('/api/sessions/', { method: 'POST' });
      const session = await res.json();
      if (session && session.id) {
        window.location.href = '/chat/' + session.id + '/';
      }
    } catch (e) {
      alert('Failed to create session: ' + e);
    }
  };

  Ada.deleteSession = async function (sessionId, event) {
    if (event) event.stopPropagation();
    if (!confirm('Delete this conversation?')) return;
    try {
      const res = await fetch('/api/sessions/' + sessionId + '/', { method: 'DELETE' });
      if (res.ok) {
        window.location.href = '/';
      }
    } catch (e) {
      alert('Failed to delete session: ' + e);
    }
  };

  Ada.renameSession = async function (sessionId, currentTitle, event) {
    if (event) event.stopPropagation();
    const newTitle = prompt('Rename conversation:', currentTitle);
    if (!newTitle || newTitle.trim() === currentTitle) return;

    try {
      const res = await fetch('/api/sessions/' + sessionId + '/', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle.trim() }),
      });
      if (res.ok) {
        window.location.reload();
      }
    } catch (e) {
      alert('Failed to rename session: ' + e);
    }
  };

  // Start polling on load
  document.addEventListener('DOMContentLoaded', () => {
    pollHealth();
  });
})();
