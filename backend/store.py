"""One SQLite file holding the file catalogue, the indexing work queue, and the
lexical (BM25) half of retrieval.

Replaces fs_scan.py. Three things live here because they're the same data viewed three
ways: what files exist, which of them still need work, and what text came out of them.
Splitting them across databases would mean keeping three things consistent for no gain.

The vector half of retrieval lives in Chroma (see retrieve.py); chunk ids are shared
between the two so results can be fused without a join across storage engines.
"""
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
    """WAL is a property of the database file, not the connection, so it is set exactly
    once here. The previous code ran `PRAGMA journal_mode=WAL` on every connect; that
    pragma needs a brief exclusive lock, so any concurrent reader turned it into
    'database is locked' — which is the crash in ada.log, and the reason the app
    'sometimes didn't work'."""
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
    # Per-connection, and lock-free to set: makes a writer wait its turn instead of
    # failing instantly when another connection holds the write lock.
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")  # safe under WAL, and much faster on bulk insert
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Catalogue + queue
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", ".venv", "venv", "dist", "build",
    "$RECYCLE.BIN", "System Volume Information", ".cache", "site-packages",
}


def scan_folder(root: str, supported: set[str]) -> dict:
    """Metadata-only walk. Never opens a file's contents, so it stays fast on a folder
    with hundreds of thousands of entries.

    A file is queued for indexing when its (size, mtime) fingerprint differs from what
    was indexed — that fingerprint, not a content hash, is what makes a re-scan cheap:
    re-hashing content would mean reading every byte of every file on every scan.
    """
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
                    continue  # vanished, locked, or a broken link — not worth failing a scan over
                seen.add(path)
                ext = os.path.splitext(name)[1].lower().lstrip(".") or "noext"
                prev = known.get(path)
                unchanged = prev and prev[0] == st.st_size and prev[1] == st.st_mtime
                # An unchanged file is left exactly as it is, including one already
                # queued — re-upserting it would churn the row and inflate the "queued"
                # count the UI shows. The exception is 'error': a rescan is the user
                # asking to try again, and the usual cause is transient (model server
                # down, file locked by another program) rather than a bad file.
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
        # Smallest first: a folder's many small files finish quickly, so the user sees
        # the index becoming useful early instead of waiting behind one 200MB PDF.
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
    """Drop chunks whose file is no longer catalogued. Cheap to run at startup and it
    keeps search from returning passages attributed to a file that no longer exists."""
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
    """Path of an already-indexed file with this content id, if any.

    doc_id is a hash of the extracted text, so two byte-identical or merely
    same-content files share it. Without this check the second copy tries to insert
    chunk ids the first already owns and dies on the primary key — which is exactly
    what 40 duplicate downloads did in a real folder.
    """
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
    """Drop a file and its chunks entirely. Used when a file has vanished between the
    scan and its turn in the queue — keeping a dead path around as a permanent error
    would make the error count grow forever and mean nothing."""
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
        # Scoped to the same root as the file counts. A global count shown next to a
        # per-folder file count reads as that folder's passages, which had an empty
        # folder claiming every chunk in the index.
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


# ---------------------------------------------------------------------------
# Chunks + lexical search
# ---------------------------------------------------------------------------


def replace_chunks(path: str, rows: list[dict]) -> None:
    """Delete-then-insert in one transaction, so a re-index of a changed file can never
    leave the old and new chunks both live."""
    conn = connect()
    try:
        conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
        if rows:
            # OR REPLACE, because a chunk id can survive its file: a run interrupted
            # between writing chunks and marking the file leaves orphans, and an
            # identical document indexed under a path that has since been deleted still
            # owns its ids. Without this, that stale row blocks the file from ever being
            # indexed again — a permanent failure caused by a crash months earlier.
            conn.executemany(
                "INSERT OR REPLACE INTO chunks (id, doc_id, path, filename, page, section, kind, text) "
                "VALUES (:id,:doc_id,:path,:filename,:page,:section,:kind,:text)",
                rows,
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


def _fts_query(text: str) -> str:
    """FTS5 treats most punctuation as syntax, so a raw user question like
    'what is P-101's limit?' is a syntax error rather than a search. Quoting each token
    turns the query into a plain OR-of-terms, which is what a BM25 pre-filter wants."""
    tokens = [t for t in "".join(c if c.isalnum() else " " for c in text).split() if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in tokens[:32])


def search_lexical(query: str, k: int = 30) -> list[tuple[str, float]]:
    """BM25 over chunk text. This is the half of retrieval that finds 'DWG-11482' and
    'ASTM A193 B7' — an embedding model is trained to generalise across surface form, so
    exact part numbers and codes are precisely what it throws away."""
    q = _fts_query(query)
    if not q:
        return []
    conn = connect()
    try:
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
    return [(r["id"], r["score"]) for r in rows]  # bm25() is negative; lower is better


def corpus_summary(limit: int = 40) -> list[dict]:
    """A cheap structural map of what's indexed. Orientation beats retrieval when the
    agent doesn't yet know what exists — this is ~300 tokens that stops it guessing."""
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
    """Every catalogued file under a root, indexed or not — what organizing works from,
    since tidying a folder cares about extensions and ages, not content."""
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
