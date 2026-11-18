import logging
import os
import sqlite3
import time
from pathlib import Path

from config import DATA_DIR

log = logging.getLogger("ada.store")

DB_PATH = DATA_DIR / "ada.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path          TEXT PRIMARY KEY,
    root          TEXT NOT NULL,
    size          INTEGER NOT NULL,
    mtime         REAL NOT NULL,
    ext           TEXT NOT NULL,
    state         TEXT NOT NULL DEFAULT 'pending',
    indexed_mtime REAL,
    doc_id        TEXT,
    pages         INTEGER DEFAULT 0,
    chunks        INTEGER DEFAULT 0,
    error         TEXT,
    updated_at    REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_files_root  ON files(root);
CREATE INDEX IF NOT EXISTS idx_files_state ON files(state);

CREATE TABLE IF NOT EXISTS chunks (
    id       TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    path     TEXT NOT NULL,
    filename TEXT NOT NULL,
    page     INTEGER NOT NULL,
    section  TEXT NOT NULL DEFAULT '',
    kind     TEXT NOT NULL DEFAULT 'text',
    text     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc  ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);

-- External-content FTS: the text lives once, in chunks. Without content=, SQLite would
-- store a second full copy of every chunk, roughly doubling the on-disk index.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, filename, section, content='chunks', content_rowid='rowid', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, filename, section)
    VALUES (new.rowid, new.text, new.filename, new.section);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, filename, section)
    VALUES ('delete', old.rowid, old.text, old.filename, old.section);
END;
"""


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


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)


    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".venv", "venv", "dist", "build",
    "$RECYCLE.BIN", "System Volume Information", ".cache", "site-packages",
}


def scan_folder(root: str, supported: set[str]) -> dict:
    root = os.path.abspath(root)
    start = time.time()
    conn = connect()
    try:
        known = {
            r["path"]: (r["size"], r["mtime"], r["state"])
            for r in conn.execute(
                "SELECT path, size, mtime, state FROM files WHERE root = ?", (root,)
            )
        }
        seen, upserts = set(), []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                except OSError:
                    continue
                seen.add(path)
                ext = os.path.splitext(name)[1].lower().lstrip(".") or "noext"
                prev = known.get(path)
                unchanged = prev and prev[0] == st.st_size and prev[1] == st.st_mtime


                if unchanged and prev[2] != "error":
                    continue
                state = "pending" if ext in supported else "unsupported"
                upserts.append((path, root, st.st_size, st.st_mtime, ext, state, time.time()))

        if upserts:
            conn.executemany(
                "INSERT INTO files (path, root, size, mtime, ext, state, updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size, mtime=excluded.mtime, ext=excluded.ext, "
                "state=excluded.state, updated_at=excluded.updated_at",
                upserts,
            )
        stale = set(known) - seen
        if stale:
            conn.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in stale])
            conn.executemany("DELETE FROM chunks WHERE path = ?", [(p,) for p in stale])
        conn.commit()
        total, pending = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(state='pending'),0) FROM files WHERE root = ?", (root,)
        ).fetchone()
    finally:
        conn.close()

    return {
        "root": root,
        "queued": len(upserts),
        "removed": len(stale),
        "total_files": total,
        "pending": pending,
        "seconds": round(time.time() - start, 2),
    }


def next_pending(limit: int = 1) -> list[sqlite3.Row]:
    conn = connect()
    try:


        return conn.execute(
            "SELECT * FROM files WHERE state='pending' ORDER BY size ASC LIMIT ?", (limit,)
        ).fetchall()
    finally:
        conn.close()


def mark(path: str, state: str, **fields) -> None:
    cols = ", ".join(f"{k}=?" for k in fields)
    sql = f"UPDATE files SET state=?, updated_at=?{', ' + cols if cols else ''} WHERE path=?"
    conn = connect()
    try:
        conn.execute(sql, (state, time.time(), *fields.values(), path))
        conn.commit()
    finally:
        conn.close()


def purge_orphan_chunks() -> int:
    conn = connect()
    try:
        n = conn.execute(
            "DELETE FROM chunks WHERE path NOT IN (SELECT path FROM files)"
        ).rowcount
        conn.commit()
    finally:
        conn.close()
    return n


def find_indexed_doc(doc_id: str, exclude_path: str) -> str | None:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT path FROM files WHERE doc_id = ? AND path != ? AND state = 'indexed' LIMIT 1",
            (doc_id, exclude_path),
        ).fetchone()
    finally:
        conn.close()
    return row["path"] if row else None


def forget(path: str) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        conn.execute("DELETE FROM files WHERE path = ?", (path,))
        conn.commit()
    finally:
        conn.close()


def queue_stats(root: str | None = None) -> dict:
    conn = connect()
    try:
        where, args = ("WHERE root = ?", (root,)) if root else ("", ())
        rows = conn.execute(
            f"SELECT state, COUNT(*) c FROM files {where} GROUP BY state", args
        ).fetchall()
        by_state = {r["state"]: r["c"] for r in rows}


        if root:
            chunks = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE path IN (SELECT path FROM files WHERE root = ?)",
                (root,),
            ).fetchone()[0]
        else:
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        conn.close()
    return {
        "pending": by_state.get("pending", 0),
        "indexed": by_state.get("indexed", 0),
        "unsupported": by_state.get("unsupported", 0),
        "error": by_state.get("error", 0),
        "skipped": by_state.get("skipped", 0),
        "total": sum(by_state.values()),
        "chunks": chunks,
    }


def _normalize_chunk_rows(path: str, rows: list[dict]) -> list[dict]:
    normalized = []
    base_name = os.path.basename(path)
    for r in rows:
        normalized.append({
            "id": r["id"],
            "doc_id": r.get("doc_id") or ("doc_" + r["id"]),
            "path": r.get("path") or path,
            "filename": r.get("filename") or base_name,
            "page": int(r.get("page", 1)),
            "section": r.get("section") or "",
            "kind": r.get("kind") or "text",
            "text": r.get("text", ""),
        })
    return normalized


def replace_chunks(path: str, rows: list[dict]) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        if rows:


            conn.executemany(
                "INSERT OR REPLACE INTO chunks (id, doc_id, path, filename, page, section, kind, text) "
                "VALUES (:id,:doc_id,:path,:filename,:page,:section,:kind,:text)",
                _normalize_chunk_rows(path, rows),
            )
        conn.commit()
    finally:
        conn.close()


def append_chunks(path: str, rows: list[dict], clear_first: bool = False) -> None:
    if clear_first:
        replace_chunks(path, rows)
        return
    if not rows:
        return
    conn = connect()
    try:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks (id, doc_id, path, filename, page, section, kind, text) "
            "VALUES (:id,:doc_id,:path,:filename,:page,:section,:kind,:text)",
            _normalize_chunk_rows(path, rows),
        )
        conn.commit()
    finally:
        conn.close()


def get_chunks(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    conn = connect()
    try:
        q = ",".join("?" * len(ids))
        return {
            r["id"]: dict(r)
            for r in conn.execute(f"SELECT * FROM chunks WHERE id IN ({q})", ids)
        }
    finally:
        conn.close()


_FTS_STOP_WORDS = {
    "what", "which", "where", "when", "who", "whom", "how", "why", "does", "did", "do",
    "the", "for", "and", "is", "are", "was", "were", "been", "being", "stated", "state",
    "regarding", "according", "with", "from", "that", "this", "these", "those", "have",
    "has", "had", "will", "would", "could", "should", "about", "into", "onto", "than"
}


def _fts_query(text: str) -> str:
    all_tokens = [t for t in "".join(c if c.isalnum() else " " for c in text).split() if len(t) > 1]
    if not all_tokens:
        return ""
    content_tokens = [t for t in all_tokens if t.lower() not in _FTS_STOP_WORDS]
    tokens = content_tokens if content_tokens else all_tokens
    return " OR ".join(f'"{t}"' for t in tokens[:32])


def search_lexical(query: str, k: int = 30, filename: str | None = None) -> list[tuple[str, float]]:
    q = _fts_query(query)
    if not q:
        return []
    conn = connect()
    try:
        if filename:
            rows = conn.execute(
                "SELECT c.id, bm25(chunks_fts, 1.0, 2.0, 1.5) AS score FROM chunks_fts "
                "JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? AND c.filename = ? ORDER BY score LIMIT ?",
                (q, filename, k),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT c.id, bm25(chunks_fts, 1.0, 2.0, 1.5) AS score FROM chunks_fts "
                "JOIN chunks c ON c.rowid = chunks_fts.rowid "
                "WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
                (q, k),
            ).fetchall()
    except sqlite3.OperationalError as e:
        log.warning("fts query failed for %r: %s", query, e)
        return []
    finally:
        conn.close()
    return [(r["id"], r["score"]) for r in rows]


def corpus_summary(limit: int = 40) -> list[dict]:
    conn = connect()
    try:
        return [
            dict(r)
            for r in conn.execute(
                "SELECT root, ext, COUNT(*) files, SUM(chunks) chunks FROM files "
                "WHERE state='indexed' GROUP BY root, ext ORDER BY files DESC LIMIT ?",
                (limit,),
            )
        ]
    finally:
        conn.close()


def list_files(root: str) -> list[dict]:
    root = os.path.abspath(root)
    conn = connect()
    try:
        return [
            {"path": r["path"], "size": r["size"], "mtime": r["mtime"], "ext": r["ext"]}
            for r in conn.execute(
                "SELECT path, size, mtime, ext FROM files WHERE root = ?", (root,)
            )
        ]
    finally:
        conn.close()


def list_indexed(root: str | None = None, limit: int = 200) -> list[dict]:
    conn = connect()
    try:
        where, args = ("AND root = ?", (root,)) if root else ("", ())
        return [
            dict(r)
            for r in conn.execute(
                f"SELECT path, ext, pages, chunks FROM files WHERE state='indexed' {where} "
                "ORDER BY updated_at DESC LIMIT ?",
                (*args, limit),
            )
        ]
    finally:
        conn.close()
