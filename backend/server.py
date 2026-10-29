"""Local HTTP API for the Ada desktop app. The Electron shell spawns this as a child
process and talks to it over 127.0.0.1 — nothing here is ever exposed beyond localhost.

Two things changed shape here versus the previous version:

  * **Indexing is no longer part of a request.** It used to run inline with a 20-second
    budget, so "add a folder" returned an answer that was mostly a lie. Now a scan queues
    work and returns immediately; a background worker drains the queue and /index/status
    reports honest progress.

  * **Chat streams.** A local model generating at ~18 tokens/second behind a blocking
    POST is indistinguishable from a hang. The agent's steps, tool calls and tokens go
    out as they happen.
"""
import asyncio
import json
import logging
import os
import tempfile
import time
import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

import chat_store
import store
from config import (
    CHAT_MODEL,
    EMBED_MODEL,
    VISION_MODEL,
    chat_client,
    setup_logging,
    tuned_chat_model,
)
from extract import SUPPORTED_EXTENSIONS
from harness import run_agent, run_agent_stream, warmup
from index import index_file, indexer
from organize import apply_moves, suggest
from permissions import grant_folder, list_granted, pick_folder, revoke_folder
from retrieve import drop_document

setup_logging()
log = logging.getLogger("ada.server")

app = FastAPI(title="Ada backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # localhost-only process; no untrusted network exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    # Chunks can outlive their file when a run is interrupted between writing them and
    # marking the file done. Left behind they are unreachable but still own their ids,
    # which blocks that document from ever being indexed again.
    orphans = store.purge_orphan_chunks()
    if orphans:
        log.info("purged %d orphaned chunks", orphans)
    indexer.start()
    # Pinning the chat model in VRAM costs nothing while idle and saves a ~15s cold load
    # on the first question. Backgrounded so a slow or absent model server can't stall
    # the port from opening.
    asyncio.get_event_loop().run_in_executor(None, warmup)


@app.on_event("shutdown")
def _shutdown() -> None:
    indexer.stop()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    log.error("unhandled error on %s: %s\n%s", request.url.path, exc, traceback.format_exc())
    return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/health")
def health():
    ok, detail = True, "ok"
    try:
        chat_client().models.list()
    except Exception as e:
        ok, detail = False, str(e)
    return {
        "status": "ok" if ok else "degraded",
        "model_backend_reachable": ok,
        "detail": detail,
        "chat_model": CHAT_MODEL,
        "vision_model": VISION_MODEL,
        "embed_model": EMBED_MODEL,
        "index": indexer.status(),
    }


# ---------------------------------------------------------------------------
# Chat sessions — the frontend only ever references a session_id; the server owns the
# transcript, so a UI reload or restart never loses history.
# ---------------------------------------------------------------------------


@app.get("/sessions")
def sessions_list():
    return {"sessions": chat_store.list_sessions()}


@app.post("/sessions")
def sessions_create():
    return chat_store.create_session()


@app.get("/sessions/{session_id}/messages")
def sessions_messages(session_id: str):
    return {"messages": chat_store.get_messages(session_id)}


@app.delete("/sessions/{session_id}")
def sessions_delete(session_id: str):
    chat_store.delete_session(session_id)
    return {"deleted": session_id}


class RenameRequest(BaseModel):
    title: str


@app.patch("/sessions/{session_id}")
def sessions_rename(session_id: str, req: RenameRequest):
    renamed = chat_store.rename_session(session_id, req.title)
    if not renamed:
        raise HTTPException(404, f"no session {session_id}")
    return renamed


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.post("/chat")
def chat(req: ChatRequest):
    """Blocking chat, kept for scripting and tests. The UI uses /chat/stream."""
    history = chat_store.get_messages(req.session_id)
    result = run_agent(req.message, history=history, session_id=req.session_id)
    chat_store.add_message(req.session_id, "user", req.message)
    chat_store.add_message(req.session_id, "assistant", result["answer"])
    return result


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    history = chat_store.get_messages(req.session_id)

    def events():
        answer = ""
        try:
            for ev in run_agent_stream(req.message, history=history, session_id=req.session_id):
                if ev["t"] == "done":
                    answer = ev["answer"]
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:
            log.exception("stream failed")
            yield f"data: {json.dumps({'t': 'error', 'message': str(e)})}\n\n"
        finally:
            # Persist even on an aborted stream, so a cancelled or crashed run still
            # leaves the user's question (and whatever was produced) in the transcript.
            chat_store.add_message(req.session_id, "user", req.message)
            if answer:
                chat_store.add_message(req.session_id, "assistant", answer)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Documents and uploads
# ---------------------------------------------------------------------------


@app.get("/ingest")
def list_uploaded():
    """Lists all files in the uploads folder and their indexing status."""
    uploads_dir = store.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    conn = store.connect()
    try:
        rows = conn.execute(
            "SELECT path, ext, state, pages, chunks, error, updated_at FROM files "
            "WHERE root = ? ORDER BY updated_at DESC",
            (str(uploads_dir),),
        ).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["filename"] = os.path.basename(d["path"])
            items.append(d)
        return {"uploads": items}
    finally:
        conn.close()


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    """Uploads documents (PDFs, Office docs, text, etc.), saves them to
    DATA_DIR/uploads, and synchronously indexes them."""
    uploads_dir = store.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    grant_folder(str(uploads_dir))

    results = []
    for upload in files:
        safe_name = os.path.basename(upload.filename or f"upload_{int(time.time())}.bin")
        dest_path = str(uploads_dir / safe_name)

        content = await upload.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        st = os.stat(dest_path)
        ext = os.path.splitext(dest_path)[1].lower().lstrip(".") or "noext"

        conn = store.connect()
        try:
            conn.execute(
                "INSERT INTO files (path, root, size, mtime, ext, state, updated_at) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
                "size=excluded.size, mtime=excluded.mtime, ext=excluded.ext, updated_at=excluded.updated_at",
                (dest_path, str(uploads_dir), st.st_size, st.st_mtime, ext, "pending", time.time()),
            )
            conn.commit()
        finally:
            conn.close()

        info = index_file(dest_path, use_ocr=True)
        store.mark(
            dest_path,
            info.get("state", "indexed"),
            pages=info.get("pages", 0),
            chunks=info.get("chunks", 0),
            error=info.get("error"),
            indexed_mtime=st.st_mtime,
            doc_id=info.get("doc_id"),
        )
        info["filename"] = safe_name
        info["path"] = dest_path
        results.append(info)
    return {"ingested": results}


@app.delete("/ingest/{filename}")
def ingest_delete(filename: str):
    """Deletes an uploaded file from disk and unindexes it from SQLite and Chroma."""
    uploads_dir = store.DATA_DIR / "uploads"
    target_path = str(uploads_dir / filename)

    conn = store.connect()
    try:
        rows = conn.execute(
            "SELECT path FROM files WHERE path = ? OR path LIKE ?",
            (target_path, f"%{filename}"),
        ).fetchall()
        matching_paths = [r["path"] for r in rows]
    finally:
        conn.close()

    if not matching_paths and os.path.exists(target_path):
        matching_paths = [target_path]

    for p in matching_paths:
        drop_document(p)
        store.forget(p)
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception as e:
                log.warning("Could not delete file %s: %s", p, e)

    if os.path.exists(target_path):
        try:
            os.remove(target_path)
        except Exception:
            pass

    # Invalidate cached eval results if they referenced this file
    from eval_rag import DEFAULT_RESULTS

    if DEFAULT_RESULTS.exists():
        try:
            with open(DEFAULT_RESULTS, "r", encoding="utf-8") as f:
                data = json.load(f)
            src = data.get("benchmark_summary", {}).get("source_pdf", "")
            if filename in src or any(p in src for p in matching_paths):
                os.remove(DEFAULT_RESULTS)
        except Exception:
            pass

    return {"deleted": filename, "paths_removed": matching_paths}


@app.get("/folders")
def folders():
    return {"granted": list_granted(), "index": indexer.status()}


@app.post("/folders/pick")
def folders_pick():
    folder = pick_folder()
    if not folder:
        return {"picked": None, "granted": list_granted()}
    return {"picked": folder, "granted": grant_folder(folder)}


class FolderRequest(BaseModel):
    folder: str


@app.post("/folders/refresh")
def folders_refresh(req: FolderRequest):
    """Queues work and returns at once. The scan itself is metadata-only, so this stays
    fast even on a folder with hundreds of thousands of files."""
    if not req.folder:
        raise HTTPException(400, "folder is required")
    if req.folder not in list_granted():
        raise HTTPException(403, f"{req.folder} was not granted to Ada")
    scan = store.scan_folder(req.folder, SUPPORTED_EXTENSIONS)
    indexer.start()
    suggestion = suggest(req.folder)
    return {
        "scan": scan,
        "index": indexer.status(),
        "suggestions": suggestion["items"],
        "suggestions_total": suggestion["total_candidates"],
        "suggestions_truncated": suggestion["truncated"],
        "dedup_skipped": suggestion["dedup_skipped"],
    }


@app.get("/index/status")
def index_status(folder: str | None = None):
    status = indexer.status()
    if folder:
        status.update(store.queue_stats(os.path.abspath(folder)))
    return status


@app.get("/index/files")
def index_files(folder: str | None = None, limit: int = 200):
    files = store.list_indexed(os.path.abspath(folder) if folder else None, limit)
    for f in files:
        f["filename"] = os.path.basename(f["path"])
    return {"files": files}


@app.delete("/folders")
def folders_revoke(folder: str):
    return {"granted": revoke_folder(folder)}


class ApplyRequest(BaseModel):
    rows: list[dict]


@app.post("/apply")
def apply_route(req: ApplyRequest):
    return apply_moves(req.rows)


# ---------------------------------------------------------------------------
# RAG Evaluation Benchmark
# ---------------------------------------------------------------------------


@app.get("/eval/rag")
def eval_rag_route(offline: bool = False, force_refresh: bool = False, pdf: str | None = None):
    """Runs or retrieves the latest RAG evaluation benchmark results measuring retrieval
    accuracy, answer correctness, faithfulness, and response time on uploaded PDFs."""
    from eval_rag import DEFAULT_RESULTS, run_rag_evaluation

    if not force_refresh and not pdf and DEFAULT_RESULTS.exists():
        try:
            with open(DEFAULT_RESULTS, "r", encoding="utf-8") as f:
                cached = json.load(f)
                if cached and cached.get("status") in ("success", "no_uploaded_pdfs"):
                    return cached
        except Exception:
            pass
    return run_rag_evaluation(pdf_path=pdf, offline=offline, quiet=True)


@app.post("/eval/rag")
def eval_rag_post_route(offline: bool = False, pdf: str | None = None):
    """Triggers a fresh execution of the RAG benchmark suite on uploaded PDFs."""
    from eval_rag import run_rag_evaluation

    return run_rag_evaluation(pdf_path=pdf, offline=offline, quiet=True)


@app.post("/eval/rag/upload")
async def eval_rag_upload_route(file: UploadFile = File(...)):
    """Uploads a PDF, indexes it, and immediately evaluates RAG on questions extracted from it."""
    from eval_rag import run_rag_evaluation

    uploads_dir = store.DATA_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    grant_folder(str(uploads_dir))

    safe_name = os.path.basename(file.filename or "uploaded.pdf")
    file_path = str(uploads_dir / safe_name)
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    st = os.stat(file_path)
    ext = os.path.splitext(file_path)[1].lower().lstrip(".") or "pdf"

    conn = store.connect()
    try:
        conn.execute(
            "INSERT INTO files (path, root, size, mtime, ext, state, updated_at) "
            "VALUES (?,?,?,?,?,?,?) ON CONFLICT(path) DO UPDATE SET "
            "size=excluded.size, mtime=excluded.mtime, ext=excluded.ext, updated_at=excluded.updated_at",
            (file_path, str(uploads_dir), st.st_size, st.st_mtime, ext, "pending", time.time()),
        )
        conn.commit()
    finally:
        conn.close()

    info = index_file(file_path, use_ocr=True)
    store.mark(
        file_path,
        info.get("state", "indexed"),
        pages=info.get("pages", 0),
        chunks=info.get("chunks", 0),
        error=info.get("error"),
        indexed_mtime=st.st_mtime,
        doc_id=info.get("doc_id"),
    )

    return run_rag_evaluation(pdf_path=file_path, quiet=True)
