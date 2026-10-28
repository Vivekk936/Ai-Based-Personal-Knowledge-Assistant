"""Persisted chat sessions and messages — SQLite, same pattern as store.py.
Keeps the frontend simple: it never manages conversation history itself, it just
references a session_id and the server keeps the transcript.

Separate from store.py's database on purpose: conversations are the user's data and the
index is a derived cache. Keeping them apart means the index can be deleted and rebuilt
without touching a single chat.

A fresh connection per call (not a shared module-level one) is deliberate: FastAPI
runs sync routes in a thread pool, and sqlite3 connections aren't safe to share
across threads without extra care — one connection per call sidesteps that
entirely, and SQLite's own connection overhead is negligible next to a network or
model call.
"""
import sqlite3
import time
import uuid
from pathlib import Path

from config import DATA_DIR

DB_PATH = DATA_DIR / "chats.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_updated ON sessions(updated_at);
"""

DEFAULT_TITLE = "New chat"
MAX_TITLE_LENGTH = 100


def _init_once() -> None:
    """Journal mode and schema are properties of the database file, so they are set once,
    at import, rather than on every connection.

    Doing this per-connect was an outright bug: `PRAGMA journal_mode=WAL` needs a brief
    exclusive lock, so any concurrent reader made it fail with 'database is locked' —
    which is the traceback in ada.log and a large part of why the app "sometimes didn't
    work". Running the full schema script on every connect was wasted work on top.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


_init_once()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    # Both are per-connection and neither takes a lock. busy_timeout makes a writer wait
    # its turn instead of failing instantly when another connection holds the write lock.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")  # makes ON DELETE CASCADE actually fire
    return conn


def create_session() -> dict:
    session_id = uuid.uuid4().hex[:12]
    now = time.time()
    conn = _connect()
    conn.execute(
        "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (session_id, DEFAULT_TITLE, now, now),
    )
    conn.commit()
    conn.close()
    return {"id": session_id, "title": DEFAULT_TITLE, "updated_at": now}


def list_sessions() -> list[dict]:
    conn = _connect()
    rows = conn.execute("SELECT id, title, updated_at FROM sessions ORDER BY updated_at DESC").fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def get_messages(session_id: str) -> list[dict]:
    conn = _connect()
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in rows]


def add_message(session_id: str, role: str, content: str) -> None:
    now = time.time()
    conn = _connect()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (session_id, role, content, now),
    )
    # First message in a session becomes its title — cheap and good enough; no
    # reason to spend a model call just to summarize a title.
    title = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
    if title and title[0] == DEFAULT_TITLE and role == "user":
        conn.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
            (content.strip().replace("\n", " ")[:MAX_TITLE_LENGTH], now, session_id),
        )
    else:
        conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))
    conn.commit()
    conn.close()


def rename_session(session_id: str, title: str) -> dict | None:
    title = title.strip()[:MAX_TITLE_LENGTH] or DEFAULT_TITLE
    conn = _connect()
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    row = conn.execute("SELECT id, title, updated_at FROM sessions WHERE id = ?", (session_id,)).fetchone()
    conn.close()
    return {"id": row[0], "title": row[1], "updated_at": row[2]} if row else None


def delete_session(session_id: str) -> None:
    conn = _connect()
    # The messages row is removed by ON DELETE CASCADE (foreign_keys=ON above) —
    # one statement, one transaction, no way for a crash to leave orphaned rows.
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
