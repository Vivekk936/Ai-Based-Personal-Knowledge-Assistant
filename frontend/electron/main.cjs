// Electron main process. Two jobs, kept deliberately separate from the UI code:
//   1. Supervise the Python backend as a child process (start it, watch it,
//      restart it with backoff if it dies, kill it cleanly on quit).
//   2. Own the app window.
// The renderer never spawns or manages the backend — it only talks to it over
// HTTP, the same as it would to any other server.
const { app, BrowserWindow } = require('electron')
const { spawn } = require('node:child_process')
const path = require('node:path')
const fs = require('node:fs')
const http = require('node:http')

const isDev = !app.isPackaged
const BACKEND_PORT = 8000
const BACKEND_DIR = path.join(__dirname, '..', '..', 'backend')
const MAX_RESTARTS = 5

let backendProcess = null
let restartCount = 0
let mainWindow = null
let quitting = false

function resolvePythonExe() {
  const candidates = [
    path.join(__dirname, '..', '..', 'backend', 'venv', 'Scripts', 'python.exe'),
    path.join(__dirname, '..', '..', '.venv', 'Scripts', 'python.exe'),
    path.join(__dirname, '..', '..', 'backend', '.venv', 'Scripts', 'python.exe'),
    path.join(__dirname, '..', '..', 'backend', 'venv', 'bin', 'python'),
    path.join(__dirname, '..', '..', '.venv', 'bin', 'python'),
  ]
  for (const c of candidates) {
    if (fs.existsSync(c)) {
      return c
    }
  }
  return process.platform === 'win32' ? 'python' : 'python3'
}

function checkBackendAlive(port = BACKEND_PORT) {
  return new Promise((resolve) => {
    const req = http.get(`http://127.0.0.1:${port}/health`, (res) => {
      resolve(res.statusCode === 200)
    })
    req.on('error', () => resolve(false))
    req.setTimeout(800, () => {
      req.destroy()
      resolve(false)
    })
  })
}

function backendLogStream() {
  const logDir = path.join(BACKEND_DIR, 'ada_data')
  fs.mkdirSync(logDir, { recursive: true })
  return fs.createWriteStream(path.join(logDir, 'backend_process.log'), { flags: 'a' })
}

async function startBackend() {
  if (quitting) return

  // 1. If backend is already running on port 8000, reuse it without spawning a conflict
  const alreadyRunning = await checkBackendAlive(BACKEND_PORT)
  if (alreadyRunning) {
    console.log(`Ada backend is already running on port ${BACKEND_PORT}. Reusing instance.`)
    mainWindow?.webContents.send('backend-status', { status: 'connected' })
    return
  }

  const pythonExe = resolvePythonExe()
  const logStream = backendLogStream()

  backendProcess = spawn(
    pythonExe,
    ['-m', 'uvicorn', 'server:app', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)],
    { cwd: BACKEND_DIR },
  )

  backendProcess.stdout.pipe(logStream)
  backendProcess.stderr.pipe(logStream)

  backendProcess.on('exit', async (code) => {
    if (quitting) return
    console.error(`Ada backend process exited (code ${code}). Checking health...`)

    // Check if another process is serving on the port
    const healthy = await checkBackendAlive(BACKEND_PORT)
    if (healthy) {
      console.log(`Port ${BACKEND_PORT} is active and healthy. Using running instance.`)
      mainWindow?.webContents.send('backend-status', { status: 'connected' })
      return
    }

    if (restartCount >= MAX_RESTARTS) {
      console.error('Backend keeps crashing — giving up after', MAX_RESTARTS, 'restarts.')
      mainWindow?.webContents.send('backend-status', { status: 'crashed' })
      return
    }

    restartCount += 1
    const delayMs = Math.min(1000 * 2 ** restartCount, 15000)
    mainWindow?.webContents.send('backend-status', { status: 'restarting', attempt: restartCount })
    setTimeout(startBackend, delayMs)
  })
}

function stopBackend() {
  quitting = true
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill()
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 860,
    title: 'Ada',
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
    },
  })

  if (isDev) {
    mainWindow.loadURL('http://localhost:5173')
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  }
}

app.whenReady().then(() => {
  startBackend()
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  stopBackend()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', stopBackend)
