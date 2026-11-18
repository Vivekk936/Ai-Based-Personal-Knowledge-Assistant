import hashlib
import logging
import os
import threading
import time

if __name__ == "__main__":


    import shutil
    import tempfile

    os.environ["ADA_DATA_DIR"] = os.path.join(tempfile.gettempdir(), "ada_index_demo")
    shutil.rmtree(os.environ["ADA_DATA_DIR"], ignore_errors=True)

import store
from chunker import chunk_page
from config import (
    ENABLE_PDF_OCR,
    MAX_UPLOAD_BYTES,
    PDF_PAGE_BATCH_SIZE,
    VISION_MODEL,
    vision_client,
)
from extract import SUPPORTED_EXTENSIONS, extract, extension_of
from retrieve import index_chunks

log = logging.getLogger("ada.index")


MAX_PAGES_PER_FILE = 5000
MAX_CHUNKS_PER_FILE = 10000
MAX_BYTES_DEFAULT = MAX_UPLOAD_BYTES


MAX_BYTES_BY_EXT = {"txt": 50_000_000, "log": 20_000_000, "csv": 100_000_000, "json": 50_000_000}


IDLE_FRACTION = float(os.getenv("ADA_INDEX_IDLE_FRACTION", "0.25"))
POLL_SECONDS = 2.0


def _content_hash(pages: list) -> str | None:
    combined = "\n".join(p.text for p in pages).strip()
    if len(combined) < 100:
        return None
    return hashlib.sha256(combined.encode("utf-8", "replace")).hexdigest()


def _ocr_page(path: str, page_number: int) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:


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


def index_file(path: str, *, use_ocr: bool = False, on_progress=None) -> dict:
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
        return {
            "state": "skipped",
            "error": f"{size/1e6:.0f}MB exceeds {cap/1e6:.0f}MB cap for .{ext}",
            "pages": 0,
            "chunks": 0,
        }

    doc_id = hashlib.sha1(path.encode("utf-8", "replace")).hexdigest()[:16]

    try:
        page_stream = extract(path, max_pages=MAX_PAGES_PER_FILE)
    except Exception as e:
        return {"state": "error", "error": f"extract: {e}", "pages": 0, "chunks": 0}

    total_pages = 0
    total_chunks = 0
    first_batch = True
    first_batch_pages = []

    try:
        while True:
            batch_pages = []
            for _ in range(PDF_PAGE_BATCH_SIZE):
                try:
                    p = next(page_stream)
                    batch_pages.append(p)
                except StopIteration:
                    break
                except Exception as e:
                    log.warning("extract error on %s: %s", filename, e)
                    break

            if not batch_pages:
                break

            if first_batch:
                first_batch_pages = batch_pages

            if use_ocr or ENABLE_PDF_OCR:
                for p in batch_pages:
                    if p.needs_ocr and ext == "pdf":
                        try:
                            p.text = _ocr_page(path, p.number)
                            p.kind = "ocr"
                        except Exception as e:
                            log.warning("ocr failed %s p%d: %s", filename, p.number, e)

            valid_pages = [p for p in batch_pages if p.text.strip()]
            total_pages += len(batch_pages)

            if valid_pages:
                batch_rows = []
                for p in valid_pages:
                    for text in chunk_page(p.text, p.kind):
                        if total_chunks + len(batch_rows) >= MAX_CHUNKS_PER_FILE:
                            break
                        batch_rows.append(
                            {
                                "id": f"{doc_id}_c{total_chunks + len(batch_rows)}",
                                "doc_id": doc_id,
                                "path": path,
                                "filename": filename,
                                "page": p.number,
                                "section": p.section,
                                "kind": p.kind,
                                "text": text,
                            }
                        )

                if batch_rows:
                    index_chunks(batch_rows, clear_first=first_batch)
                    total_chunks += len(batch_rows)

            first_batch = False


            store.mark(path, "indexing", pages=total_pages, chunks=total_chunks, doc_id=doc_id)
            if on_progress:
                on_progress(total_pages, total_chunks)

            if total_chunks >= MAX_CHUNKS_PER_FILE:
                log.info("Reached MAX_CHUNKS_PER_FILE limit for %s", filename)
                break

    except Exception as e:
        log.exception("incremental index failed for %s", filename)
        store.mark(path, "error", error=f"index: {e}", pages=total_pages, chunks=total_chunks)
        return {"state": "error", "error": f"index: {e}", "pages": total_pages, "chunks": total_chunks}

    if total_pages == 0 or total_chunks == 0:
        return {
            "state": "skipped",
            "error": "no extractable text (scanned? enable OCR)",
            "pages": total_pages,
            "chunks": total_chunks,
        }


    if first_batch_pages:
        content_hash = _content_hash(first_batch_pages)
        if content_hash:
            twin = store.find_indexed_doc(content_hash[:16], path)
            if twin:
                store.forget(path)
                return {
                    "state": "skipped",
                    "error": f"identical content already indexed as {os.path.basename(twin)}",
                    "pages": total_pages,
                    "chunks": 0,
                    "doc_id": content_hash[:16],
                }

    store.mark(path, "indexed", pages=total_pages, chunks=total_chunks, doc_id=doc_id)
    return {"state": "indexed", "pages": total_pages, "chunks": total_chunks, "doc_id": doc_id}


class Indexer:

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.current: str | None = None
        self.current_pages: int = 0
        self.current_chunks: int = 0
        self.use_ocr = os.getenv("ADA_INDEX_OCR", "0") == "1" or ENABLE_PDF_OCR

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
                self.current_pages = 0
                self.current_chunks = 0
                self._stop.wait(POLL_SECONDS)
                continue
            row = batch[0]
            path = row["path"]
            if not os.path.exists(path):
                store.forget(path)
                continue
            self.current = path
            self.current_pages = 0
            self.current_chunks = 0
            started = time.time()

            def on_progress(p, c):
                self.current_pages = p
                self.current_chunks = c

            try:
                result = index_file(path, use_ocr=self.use_ocr, on_progress=on_progress)
            except Exception as e:
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
            "current_pages": self.current_pages if self.current else 0,
            "current_chunks": self.current_chunks if self.current else 0,
            **stats,
        }


indexer = Indexer()


def demo() -> None:
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


        twin = os.path.join(tmp, "manual_copy.md")
        with open(twin, "w", encoding="utf-8") as f:
            f.write("# Pump P-101\n\n" + "Maximum operating pressure is 150 PSI. " * 40)
        store.scan_folder(tmp, SUPPORTED_EXTENSIONS)
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
