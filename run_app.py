#!/usr/bin/env python3
"""Unified Startup Script for Ada: Document Intelligence Agent.

Launches:
  1. FastAPI Backend (http://127.0.0.1:8000)
  2. Django Frontend (http://127.0.0.1:8080)
  3. Native Desktop Shell (Electron) or opens default Web Browser

Usage:
  python run_app.py              # Starts backend + Django + launches Electron (or opens browser)
  python run_app.py --web        # Starts backend + Django and opens browser directly
  python run_app.py --no-browser # Starts backend + Django in background without launching UI
"""
import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
DJANGO_DIR = ROOT_DIR / "django_frontend"
ELECTRON_DIR = ROOT_DIR / "frontend"

BACKEND_PORT = 8000
DJANGO_PORT = 8080

processes: list[subprocess.Popen] = []


def is_port_active(port: int, path: str = "/") -> bool:
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=1.0) as resp:
            return resp.status < 500
    except Exception:
        return False


def wait_for_port(port: int, path: str = "/", timeout: float = 20.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if is_port_active(port, path):
            return True
        time.sleep(0.5)
    return False


def cleanup(*_):
    print("\n[Ada] Shutting down services cleanly...")
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    print("[Ada] All services terminated.")
    sys.exit(0)


def resolve_python() -> str:
    venv_py = BACKEND_DIR / "venv" / ("Scripts" if sys.platform == "win32" else "bin") / ("python.exe" if sys.platform == "win32" else "python")
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main():
    parser = argparse.ArgumentParser(description="Ada Document Intelligence Agent Launcher")
    parser.add_argument("--web", action="store_true", help="Launch in web browser instead of Electron")
    parser.add_argument("--no-browser", action="store_true", help="Do not open any client window")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    py_exe = resolve_python()

    # 1. Start FastAPI Backend if not already running
    if is_port_active(BACKEND_PORT, "/health"):
        print(f"[Ada Backend] Already active on port {BACKEND_PORT}. Reusing instance.")
    else:
        print(f"[Ada Backend] Starting FastAPI on 127.0.0.1:{BACKEND_PORT}...")
        backend_p = subprocess.Popen(
            [py_exe, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            cwd=str(BACKEND_DIR),
        )
        processes.append(backend_p)
        if not wait_for_port(BACKEND_PORT, "/health", timeout=25.0):
            print("[Ada Backend] Warning: backend did not report healthy within timeout.")

    # 2. Start Django Frontend if not already running
    if is_port_active(DJANGO_PORT, "/api/health/"):
        print(f"[Ada Django] Already active on port {DJANGO_PORT}. Reusing instance.")
    else:
        print(f"[Ada Django] Starting Django Frontend on 127.0.0.1:{DJANGO_PORT}...")
        django_p = subprocess.Popen(
            [py_exe, "manage.py", "runserver", f"127.0.0.1:{DJANGO_PORT}", "--noreload"],
            cwd=str(DJANGO_DIR),
        )
        processes.append(django_p)
        if not wait_for_port(DJANGO_PORT, "/api/health/", timeout=25.0):
            print("[Ada Django] Warning: Django frontend did not respond within timeout.")

    url = f"http://127.0.0.1:{DJANGO_PORT}"
    print(f"\n=======================================================")
    print(f"  Ada is running:")
    print(f"  - Django Frontend UI:  {url}")
    print(f"  - FastAPI Backend API: http://127.0.0.1:{BACKEND_PORT}")
    print(f"=======================================================\n")

    if args.no_browser:
        print("[Ada] Running in headless/service mode. Press Ctrl+C to terminate.")
    elif args.web:
        print(f"[Ada] Opening default browser at {url}...")
        webbrowser.open(url)
    else:
        # Check if electron is available
        electron_main = ELECTRON_DIR / "electron" / "main.cjs"
        electron_node_modules = ELECTRON_DIR / "node_modules"
        has_electron = electron_main.exists() and electron_node_modules.exists()

        if has_electron:
            print("[Ada Desktop] Launching Electron desktop window...")
            try:
                npm_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
                electron_p = subprocess.Popen(
                    [npm_cmd, "electron", "."],
                    cwd=str(ELECTRON_DIR),
                )
                processes.append(electron_p)
                electron_p.wait()
                cleanup()
                return
            except Exception as e:
                print(f"[Ada Desktop] Could not start Electron ({e}). Falling back to browser.")
                webbrowser.open(url)
        else:
            print("[Ada] Opening browser UI...")
            webbrowser.open(url)

    # Keep running until Ctrl+C
    try:
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    # One of the processes died
                    print(f"[Ada] Warning: A child process exited unexpectedly (code {p.returncode}).")
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
