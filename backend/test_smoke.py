"""End-to-end checks for the scan -> index -> retrieve -> cite path.

Deliberately covers the three things that were actually broken rather than the happy path
that always worked:

  * a document larger than the old 1MB cap is indexed, not silently skipped
  * an exact part number is retrievable, which dense-only retrieval could not do
  * a fabricated citation is caught by code, not trusted

Run directly: `..\\.venv\\Scripts\\python.exe test_smoke.py`. Anything needing the
embedding server is skipped with a clear message when it isn't reachable, so the
deterministic half still runs offline.
"""
import ctypes
import os
import shutil
import tempfile

# Its own database, set before anything imports config. The smoke test indexes documents
# and asserts on what comes back top-ranked; sharing the real index means whatever the
# user happens to have indexed decides whether the test passes.
os.environ.setdefault("ADA_DATA_DIR", os.path.join(tempfile.gettempdir(), "ada_smoke_data"))
shutil.rmtree(os.environ["ADA_DATA_DIR"], ignore_errors=True)

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

import store
from chunker import chunk_page
from harness import _call_tool, _unsupported_numbers, _verify_citations
from index import index_file
from permissions import grant_folder, revoke_folder

PART_NUMBER = "DWG-11482-C"
PRESSURE = "150 PSI"


def make_pdf(path: str, lines: list[str], pad_to_bytes: int = 0) -> None:
    doc = pdfium.PdfDocument.new()
    font = pdfium_c.FPDFText_LoadStandardFont(doc, b"Helvetica")
    for line in lines:
        page = doc.new_page(595, 842)
        obj = pdfium_c.FPDFPageObj_CreateTextObj(doc, font, 12)
        buf = ctypes.create_string_buffer(line.encode("utf-16-le") + b"\x00\x00")
        pdfium_c.FPDFText_SetText(obj, ctypes.cast(buf, pdfium_c.FPDF_WIDESTRING))
        pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 50, 700)
        pdfium_c.FPDFPage_InsertObject(page, obj)
        pdfium_c.FPDFPage_GenerateContent(page)
    doc.save(path)
    doc.close()
    if pad_to_bytes:
        # PDF readers ignore trailing bytes after %%EOF, so this inflates the file past
        # the old size cap without disturbing the parsed content.
        with open(path, "ab") as f:
            f.write(b"\n% " + b"x" * max(0, pad_to_bytes - os.path.getsize(path)))


def embedding_server_up() -> bool:
    from config import embed_texts

    try:
        embed_texts(["ping"])
        return True
    except Exception:
        return False


def test_deterministic() -> None:
    """No model server required."""
    assert _verify_citations("Pressure is 150 PSI [m.pdf:2].", {("m.pdf", "2")}) == []
    assert _verify_citations("Invented [ghost.pdf:9].", {("m.pdf", "2")}) == ["[ghost.pdf:9]"]

    ev = f"The drawing {PART_NUMBER} specifies {PRESSURE} at the flange."
    assert _unsupported_numbers(f"It is rated to {PRESSURE}.", ev) == []
    assert _unsupported_numbers("It is rated to 900 PSI.", ev) == ["900 PSI"]

    assert "No tool named" in _call_tool("search_docs", "{}")
    assert "Could not parse" in _call_tool("search_documents", "{broken")

    big_table = "Part | Qty\n" + "\n".join(f"{PART_NUMBER}-{i} | {i}" for i in range(200))
    parts = chunk_page(big_table, kind="table")
    assert len(parts) > 1 and all(p.startswith("Part | Qty") for p in parts)
    print("PASS deterministic — citation + number verification, tool errors, table chunking")


def test_pipeline() -> None:
    if not embedding_server_up():
        print("SKIP pipeline — embedding server unreachable (check NVIDIA_API_KEY in .env to run this)")
        return

    tmp = tempfile.mkdtemp(prefix="ada_smoke_")
    try:
        grant_folder(tmp)

        big_pdf = os.path.join(tmp, "manual.pdf")
        make_pdf(
            big_pdf,
            [
                f"Pump P-101 maximum operating pressure is {PRESSURE} at the discharge flange.",
                f"Refer to drawing {PART_NUMBER} for the bearing housing detail.",
            ],
            pad_to_bytes=1_500_000,  # over the old MAX_INDEXABLE_BYTES of 1,000,000
        )
        assert os.path.getsize(big_pdf) > 1_000_000

        result = index_file(big_pdf)
        assert result["state"] == "indexed", result
        assert result["chunks"] > 0, result
        print(f"PASS large file — {os.path.getsize(big_pdf)/1e6:.1f}MB PDF indexed "
              f"({result['pages']} pages, {result['chunks']} chunks); old cap skipped this")

        lexical = store.search_lexical(PART_NUMBER, k=5)
        assert lexical, f"BM25 must find the exact part number {PART_NUMBER}"
        found = store.get_chunks([cid for cid, _ in lexical])
        assert any(PART_NUMBER in c["text"] for c in found.values()), list(found.values())
        print(f"PASS exact match — BM25 retrieved {PART_NUMBER} verbatim")

        from retrieve import confidence, search

        hits = search(f"what pressure is pump P-101 rated for", k=5)
        assert hits, "hybrid search returned nothing"
        assert any(PRESSURE in h["text"] for h in hits), [h["text"][:80] for h in hits]
        assert hits[0]["filename"] == "manual.pdf"
        print(f"PASS hybrid — semantic question answered, confidence={confidence(hits)}")

        scan = store.scan_folder(tmp, {"pdf"})
        assert scan["total_files"] >= 1
        again = store.scan_folder(tmp, {"pdf"})
        assert again["queued"] == 0, f"unchanged files must not requeue: {again}"
        print("PASS incremental — rescan queues nothing when nothing changed")

        from tools import read_chunk, search_documents

        out = search_documents("pump pressure rating", k=3)
        assert "id=" in out and "manual.pdf" in out, out
        cid = out.split("id=")[1].split()[0]
        full = read_chunk(cid)
        assert "<document_text>" in full and "manual.pdf" in full
        assert "No chunk with id" in read_chunk("nonexistent_p1_c0")
        print("PASS tools — two-hop search->read_chunk with citable headers")

    finally:
        revoke_folder(tmp)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_deterministic()
    test_pipeline()
    print("\nAll smoke checks passed.")
