// Bridges main-process backend-status events to the renderer. Deliberately the
// only thing exposed — the renderer has no Node/filesystem access, it only
// gets to know "is the backend alive" and talks to it over plain HTTP.
const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('ada', {
  onBackendStatus: (callback) => {
    ipcRenderer.on('backend-status', (_event, status) => callback(status))
  },
  backendUrl: 'http://127.0.0.1:8000',
})
