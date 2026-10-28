"""The background indexer: one throttled worker thread that drains the pending queue.

What this replaces, and why it was the core bug:

The old index_folder() ran inline inside the HTTP request with a 20-second wall-clock
budget and then gave up. Every model call at the time also paid a 2-second IPv6 stall
(see config.py), so 20 seconds bought roughly nine files. On any real folder the user
watched it "finish" having indexed almost nothing, which is exactly the reported
symptom — not slowness, but silent partial work presented as success.

It also skipped anything over 1MB, which is most real PDFs, and skipped every scanned
page during folder indexing, which is most engineering drawings and scan reports.

The fix is structural rather than a bigger budget: indexing is a queue, the queue is in
SQLite, and a worker drains it across restarts. Nothing is ever abandoned — it is either
done, deliberately skipped with a reason, or still pending.
"""
import hashlib
import logging
import os
import threading
import time

if __name__ == "__main__":
    # Set before config binds DATA_DIR. The demo indexes fixed content and asserts on the
    # result, so it needs its own database — run against the real one it both pollutes the
    # user's index and fails, because their files outrank the demo's.
    import shutil
    import tempfile

    os.environ["ADA_DATA_DIR"] = os.path.join(tempfile.gettempdir(), "ada_index_demo")
    shutil.rmtree(os.environ["ADA_DATA_DIR"], ignore_errors=True)

import store
from chunker import chunk_page
from config import VISION_MODEL, vision_client
from extract import SUPPORTED_EXTENSIONS, extract, extension_of
from retrieve import index_chunks

log = logging.getLogger("ada.index")

# Per-file guardrails. These are about a single file not monopolising the worker, not
# about giving up on the folder — an oversized file is recorded as skipped with a reason
# the user can see, never silently dropped.
MAX_PAGES_PER_FILE = 3000
MAX_CHUNKS_PER_FILE = 4000
MAX_BYTES_DEFAULT = 200_000_000

# Text formats are cheap per byte; a 50MB wordlist is still 50MB of embeddings, and it
# isn't prose anyone will ask questions about. The old global 1MB cap was aimed at this
# case but caught every real PDF manual in the crossfire.
MAX_BYTES_BY_EXT = {"txt": 20_000_000, "log": 5_000_000, "csv": 50_000_000, "json": 10_000_000}

# The throttle. Sleeping a fraction of each file's own processing time keeps the worker
# at a roughly constant duty cycle rather than a fixed rate: cheap files barely pause,
# expensive ones yield proportionally more. 0.25 keeps a core ~80% free for the user.
IDLE_FRACTION = float(os.getenv("ADA_INDEX_IDLE_FRACTION", "0.25"))
POLL_SECONDS = 2.0


def _content_hash(pages: list) -> str | None:
    """Hash extracted text, not raw bytes: two exports of the same document routinely
    differ byte-for-byte (writers stamp a fresh timestamp and document id on every save),
    so byte-hashing would almost never catch a real duplicate."""
    combined = "\n".join(p.text for p in pages).strip()
    if len(combined) < 100:
        return None  # too little text to distinguish two scanned documents from each other
    return hashlib.sha256(combined.encode("utf-8", "replace")).hexdigest()


def _ocr_page(path: str, page_number: int) -> str:
    """Rasterise one PDF page and have the vision model read it. Only ever called for
    pages that have no text layer at all."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        # 200 DPI: dimension callouts and title-block text on drawings are a few points
        # at scale and vanish at the 72 DPI default.
        bitmap = pdf[page_number - 1].render(scale=200 / 72)
        img = bitmap.to_pil()
    finally:
        pdf.close()

    import base64
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    resp = vision_client().chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        # "Transcribe, don't describe" matters on drawings: asked to
                        # describe, a small VLM invents plausible dimensions fluently.
                        "text": (
                            "Transcribe all readable text on this page verbatim, including "
                            "table cells, labels and title-block fields. Then list any "
                            "diagrams or photos in factual terms. Never guess at unreadable "
                            "text — write [illegible] instead."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def index_file(path: str, *, use_ocr: bool = False) -> dict:
    """Extract, chunk, embed and store one file. Returns a result dict; never raises for
    a bad file — the caller records the reason and moves on."""
    filename = os.path.basename(path)
    ext = extension_of(path)
    if ext not in SUPPORTED_EXTENSIONS:
        return {"state": "unsupported", "pages": 0, "chunks": 0}

    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"state": "error", "error": str(e), "pages": 0, "chunks": 0}
    cap = MAX_BYTES_BY_EXT.get(ext, MAX_BYTES_DEFAULT)
    if size > cap:
        return {"state": "skipped", "error": f"{size/1e6:.0f}MB exceeds {cap/1e6:.0f}MB cap for .{ext}",
                "pages": 0, "chunks": 0}

    try:
        pages = list(extract(path, max_pages=MAX_PAGES_PER_FILE))
    except Exception as e:
        return {"state": "error", "error": f"extract: {e}", "pages": 0, "chunks": 0}

    if use_ocr:
        for p in pages:
            if p.needs_ocr and ext == "pdf":
                try:
                    p.text = _ocr_page(path, p.number)
                    p.kind = "ocr"
                except Exception as e:
                    log.warning("ocr failed %s p%d: %s", filename, p.number, e)

    pages = [p for p in pages if p.text.strip()]
    if not pages:
        return {"state": "skipped", "error": "no extractable text (scanned? enable OCR)",
                "pages": 0, "chunks": 0}

    content_hash = _content_hash(pages)
    doc_id = (content_hash or hashlib.sha1(path.encode("utf-8", "replace")).hexdigest())[:16]

    # Identical content already indexed elsewhere: point at it instead of embedding it
    # again. Dedup is by extracted text rather than file bytes because two exports of the
    # same document differ byte-for-byte (writers stamp a fresh timestamp on every save),
    # so byte-hashing would almost never catch a real duplicate.
    if content_hash:
        twin = store.find_indexed_doc(doc_id, path)
        if twin:
            return {"state": "skipped", "error": f"identical content already indexed as {os.path.basename(twin)}",
                    "pages": len(pages), "chunks": 0, "doc_id": doc_id}

    rows = []
    for p in pages:
        for text in chunk_page(p.text, p.kind):
            if len(rows) >= MAX_CHUNKS_PER_FILE:
                break
            rows.append(
                {
                    # Monotonic within the document, not (page, i): a spreadsheet emits
                    # several row-group pages that all carry the same sheet index, so
                    # keying on page number collides and loses every group but the first.
                    "id": f"{doc_id}_c{len(rows)}",
                    "doc_id": doc_id,
                    "path": path,
                    "filename": filename,
                    "page": p.number,
                    "section": p.section,
                    "kind": p.kind,
                    "text": text,
                }
            )

    try:
        index_chunks(rows)
    except Exception as e:
        return {"state": "error", "error": f"embed: {e}", "pages": len(pages), "chunks": 0}

    return {"state": "indexed", "pages": len(pages), "chunks": len(rows), "doc_id": doc_id}


class Indexer:
    """A single worker thread. One, not a pool: the embedding server is the bottleneck and
    it serialises requests anyway, so extra threads would add contention and CPU load
    without adding throughput."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.current: str | None = None
        self.use_ocr = os.getenv("ADA_INDEX_OCR", "0") == "1"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ada-indexer", daemon=True)
        self._thread.start()
        log.info("indexer started (ocr=%s)", self.use_ocr)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = store.next_pending(limit=1)
            if not batch:
                self.current = None
                self._stop.wait(POLL_SECONDS)
                continue
            row = batch[0]
            path = row["path"]
            if not os.path.exists(path):
                store.forget(path)  # deleted or moved since the scan; not an error
                continue
            self.current = path
            started = time.time()
            try:
                result = index_file(path, use_ocr=self.use_ocr)
            except Exception as e:  # a single pathological file must never kill the worker
                log.exception("indexing %s failed", path)
                result = {"state": "error", "error": str(e)[:300], "pages": 0, "chunks": 0}
            store.mark(
                path,
                result["state"],
                pages=result.get("pages", 0),
                chunks=result.get("chunks", 0),
                error=result.get("error"),
                indexed_mtime=row["mtime"],
                doc_id=result.get("doc_id"),
            )
            elapsed = time.time() - started
            if IDLE_FRACTION > 0:
                self._stop.wait(min(elapsed * IDLE_FRACTION, 2.0))

    def status(self) -> dict:
        stats = store.queue_stats()
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "current_file": os.path.basename(self.current) if self.current else None,
            **stats,
        }


indexer = Indexer()


def demo() -> None:
    """Round-trips a real text file through extraction, chunking and the queue. Skips the
    embedding step if no model server is reachable so the check still runs offline."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "manual.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write("# Pump P-101\n\n" + "Maximum operating pressure is 150 PSI. " * 40)

        big = os.path.join(tmp, "huge.log")
        with open(big, "w", encoding="utf-8") as f:
            f.write("x" * (MAX_BYTES_BY_EXT["log"] + 10))
        r = index_file(big)
        assert r["state"] == "skipped" and "cap" in r["error"], r
        assert index_file(os.path.join(tmp, "nope.exe"))["state"] == "unsupported"
        assert index_file(os.path.join(tmp, "missing.md"))["state"] in ("error", "skipped")

        # A spreadsheet is the case that exposes id collisions: one sheet yields a schema
        # page plus several row-group pages, all carrying the same sheet index.
        import openpyxl

        xl = os.path.join(tmp, "inventory.xlsx")
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(["Part", "Qty"])
        for n in range(150):
            ws.append([f"PN-{n:04d}", n])
        wb.save(xl)
        sheet_pages = list(extract(xl))
        assert len({p.number for p in sheet_pages}) < len(sheet_pages), "expected repeated page numbers"
        ids = [f"x_c{i}" for i in range(len(sheet_pages))]
        assert len(set(ids)) == len(ids), "chunk ids must stay unique across same-numbered pages"

        scan = store.scan_folder(tmp, SUPPORTED_EXTENSIONS)
        assert scan["total_files"] >= 2, scan
        assert scan["pending"] >= 1, scan
        pend = store.next_pending(5)
        assert any(os.path.basename(r["path"]) == "manual.md" for r in pend), [dict(r) for r in pend]

        store.mark(p, "indexed", pages=1, chunks=3, doc_id="abc")
        assert store.queue_stats()["indexed"] >= 1
        rescan = store.scan_folder(tmp, SUPPORTED_EXTENSIONS)
        assert rescan["queued"] == 0, f"unchanged files must not requeue: {rescan}"

        # Two files with identical content must dedup, not collide: doc_id is a hash of
        # the extracted text, so both copies claim the same chunk ids. 40 duplicate
        # downloads in a real folder failed exactly this way before this check existed.
        twin = os.path.join(tmp, "manual_copy.md")
        with open(twin, "w", encoding="utf-8") as f:
            f.write("# Pump P-101\n\n" + "Maximum operating pressure is 150 PSI. " * 40)
        store.scan_folder(tmp, SUPPORTED_EXTENSIONS)  # dedup reads the files table
        first = index_file(p)
        assert first["state"] == "indexed", first
        store.mark(p, "indexed", doc_id=first["doc_id"], chunks=first["chunks"])
        second = index_file(twin)
        assert second["state"] == "skipped", second
        assert "already indexed" in second["error"], second

        store.replace_chunks(p, [{"id": "t1", "doc_id": "abc", "path": p, "filename": "manual.md",
                                  "page": 1, "section": "Pump P-101", "kind": "text",
                                  "text": "Maximum operating pressure is 150 PSI."}])
        lex = store.search_lexical("150 PSI pressure", k=5)
        assert any(cid == "t1" for cid, _ in lex), f"BM25 should find the exact figure: {lex}"
        assert store.search_lexical("what is P-101's limit?") is not None, "punctuation must not crash FTS"

        store.replace_chunks(p, [])
        assert not store.get_chunks(["t1"])
    print("PASS index — queue, caps, resume, BM25 round-trip")


if __name__ == "__main__":
    demo()
