"""Folder access allow-list. Ada only ever touches paths under folders the user
explicitly granted here — nothing is scanned, read, or moved without an explicit grant.
"""
import json
import os
from pathlib import Path

from config import DATA_DIR

SETTINGS_PATH = DATA_DIR / "settings.json"


def _load() -> dict:
    if SETTINGS_PATH.exists():
        return json.loads(SETTINGS_PATH.read_text())
    return {"granted_folders": []}


def _save(data: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2))


def list_granted() -> list[str]:
    granted = list(_load()["granted_folders"])
    uploads_dir = os.path.abspath(str(DATA_DIR / "uploads"))
    if uploads_dir not in granted and os.path.exists(uploads_dir):
        granted.append(uploads_dir)
    return granted


def grant_folder(path: str) -> list[str]:
    path = os.path.abspath(path)
    data = _load()
    if path not in data["granted_folders"]:
        data["granted_folders"].append(path)
        _save(data)
    return data["granted_folders"]


def revoke_folder(path: str) -> list[str]:
    path = os.path.abspath(path)
    data = _load()
    data["granted_folders"] = [f for f in data["granted_folders"] if f != path]
    _save(data)
    return data["granted_folders"]


def is_allowed(path: str) -> bool:
    path = os.path.abspath(path)
    uploads_dir = os.path.abspath(str(DATA_DIR / "uploads"))
    if path == uploads_dir or path.startswith(uploads_dir + os.sep):
        return True
    return any(path == f or path.startswith(f + os.sep) for f in list_granted())


def pick_folder() -> str | None:
    """Native OS folder picker. Runs on the same machine as the server, so this
    works even though the UI itself is served over HTTP."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Grant Ada access to a folder")
    root.destroy()
    return folder
