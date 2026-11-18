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


    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
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


    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
