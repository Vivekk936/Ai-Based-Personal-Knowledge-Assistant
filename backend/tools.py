import json
import logging
import os
import re

import deliver
import store
from config import OUTPUT_DIR
from permissions import is_allowed, list_granted
from retrieve import confidence, search

log = logging.getLogger("ada.tools")

MAX_SNIPPET = 160
MAX_CHUNK_CHARS = 3000
MAX_FILE_LINES = 200
MAX_FILE_CHARS = 6000
MAX_GREP_HITS = 40
MAX_DIR_ENTRIES = 200


def _denied(path: str) -> str:
    roots = list_granted()
    return (
        f"Access denied: {path} is outside the folders you may read.\n"
        f"Granted roots: {', '.join(roots) if roots else '(none yet — ask the user to add a folder)'}\n"
        "Use list_folder on a granted root to see what is available."
    )


def search_documents(query: str, k: int = 6, filename: str = "") -> str:
    k = max(1, min(int(k or 6), 15))
    fn = filename.strip() or None
    hits = search(query, k=k, filename=fn)
    if not hits:
        stats = store.queue_stats()
        if stats["indexed"] == 0:
            return (
                "No documents are indexed yet. "
                f"{stats['pending']} files are still queued. "
                "Tell the user indexing is still in progress rather than guessing an answer."
            )
        target_doc = f" in {fn}" if fn else ""
        return (
            f"No matches for {query!r}{target_doc} across {stats['indexed']} indexed files.\n"
            "Try different wording, alternative keywords, or call list_folder to see what exists. "
            "If the corpus genuinely does not cover this, say so — do not guess."
        )

    conf = confidence(hits)
    lines = [f"Retrieval confidence: {conf}. {len(hits)} results (snippets — call read_chunk or read_chunks for full text):"]
    for h in hits:
        snippet = " ".join(h["text"].split())[:MAX_SNIPPET]
        both = "both" if h["agreed"] else ("keyword" if h["dense_rank"] is None else "semantic")
        lines.append(f"- id={h['id']} [{h['filename']}:{h['page']}] ({both}) {snippet}…")
    if conf in ("low", "none"):
        lines.append(
            "Confidence is low: nothing matched strongly. Try another search with alternative keywords before concluding."
        )
    return "\n".join(lines)


def read_chunks(chunk_ids: list[str] | str) -> str:
    if isinstance(chunk_ids, str):
        ids = [c.strip() for c in re.split(r"[,;\s]+", chunk_ids) if c.strip()]
    elif isinstance(chunk_ids, list):
        ids = [str(c).strip() for c in chunk_ids if str(c).strip()]
    else:
        return "chunk_ids must be a list of chunk IDs or a comma-separated string."

    if not ids:
        return "No chunk IDs provided."

    ids = ids[:8]
    records = store.get_chunks(ids)
    if not records:
        return f"None of the requested chunk IDs ({', '.join(ids)}) were found in the index."

    out = []
    for cid in ids:
        rec = records.get(cid)
        if not rec:
            out.append(f"[{cid}]: Not found in index.")
            continue
        text = rec["text"]
        note = ""
        if len(text) > MAX_CHUNK_CHARS:
            note = f"\n…[truncated; passage is {len(text)} chars]"
            text = text[:MAX_CHUNK_CHARS]
        head = f"[{rec['filename']}:{rec['page']}]"
        if rec["section"]:
            head += f" ({rec['section']})"
        out.append(f"{head}\n<document_text>\n{text}{note}\n</document_text>")

    return "\n\n".join(out)


def read_chunk(chunk_id: str) -> str:
    if not chunk_id:
        return "chunk_id is required."
    cids = [c.strip() for c in re.split(r"[,;\s]+", str(chunk_id)) if c.strip()]
    if len(cids) > 1:
        return read_chunks(cids)

    rec = store.get_chunks([chunk_id]).get(chunk_id)
    if not rec:
        return (
            f"No chunk with id {chunk_id!r}. Ids come from search_documents results and "
            "look like 'a1b2c3d4_c0'. Call search_documents first."
        )
    text = rec["text"]
    note = ""
    if len(text) > MAX_CHUNK_CHARS:
        note = f"\n…[truncated; passage is {len(text)} chars]"
        text = text[:MAX_CHUNK_CHARS]
    head = f"[{rec['filename']}:{rec['page']}]"
    if rec["section"]:
        head += f" ({rec['section']})"
    return f"{head}\n<document_text>\n{text}{note}\n</document_text>"


def list_folder(path: str = "", pattern: str = "") -> str:
    roots = list_granted()
    if not path:
        if not roots:
            return "No folders have been granted yet. Ask the user to add one in the app."
        return "Granted root folders:\n" + "\n".join(f"- {r}" for r in roots)
    path = os.path.abspath(path)
    if not is_allowed(path):
        return _denied(path)
    if not os.path.isdir(path):
        return f"{path} is not a folder. Use read_file for a file, or list_folder on its parent."

    try:
        entries = list(os.scandir(path))
    except OSError as e:
        return f"Could not read {path}: {e}"

    dirs, files = [], []
    for e in entries:
        if e.name.startswith(".") or e.name in store.SKIP_DIRS:
            continue
        if pattern and not re.search(pattern, e.name, re.I):
            continue
        try:
            (dirs if e.is_dir() else files).append(
                (e.name, 0 if e.is_dir() else e.stat().st_size)
            )
        except OSError:
            continue

    total = len(dirs) + len(files)
    rows = [f"[dir]  {n}/" for n, _ in sorted(dirs)]
    rows += [f"       {n}  ({s:,} bytes)" for n, s in sorted(files)]
    out = "\n".join(rows[:MAX_DIR_ENTRIES])
    if total > MAX_DIR_ENTRIES:
        out += f"\n…[{total} entries total; showing {MAX_DIR_ENTRIES}. Use `pattern` to narrow.]"
    return f"{path} — {len(dirs)} folders, {len(files)} files\n{out}" if rows else f"{path} is empty."


def read_file(path: str, offset: int = 0, limit: int = MAX_FILE_LINES) -> str:
    path = os.path.abspath(path)
    if not is_allowed(path):
        return _denied(path)
    if not os.path.isfile(path):
        return f"{path} is not a file. Use list_folder to see what exists."

    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or MAX_FILE_LINES), 400))
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError as e:
        return f"Could not read {path}: {e}"

    total = len(lines)
    window = lines[offset : offset + limit]
    body = "".join(window)
    if len(body) > MAX_FILE_CHARS:
        body = body[:MAX_FILE_CHARS] + "\n…[line too long, truncated]"
    shown_to = offset + len(window)
    footer = ""
    if shown_to < total:
        footer = f"\n…[{total} lines total; showing {offset + 1}–{shown_to}. Call read_file with offset={shown_to} for more.]"
    return f"{path} ({total} lines)\n<file_text>\n{body}\n</file_text>{footer}"


def grep_files(pattern: str, path: str = "", max_results: int = MAX_GREP_HITS) -> str:
    roots = [os.path.abspath(path)] if path else list_granted()
    if path and not is_allowed(roots[0]):
        return _denied(path)
    if not roots:
        return "No folders granted yet, so there is nothing to search."
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        return f"Invalid regex {pattern!r}: {e}. Use a plain substring if unsure."

    max_results = max(1, min(int(max_results or MAX_GREP_HITS), 100))
    hits, scanned, capped = [], 0, False
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in store.SKIP_DIRS]
            for name in filenames:
                if len(hits) >= max_results:
                    capped = True
                    break
                fp = os.path.join(dirpath, name)
                if os.path.splitext(name)[1].lower().lstrip(".") not in _TEXTY:
                    continue
                scanned += 1
                try:
                    with open(fp, "r", encoding="utf-8", errors="replace") as f:
                        for n, line in enumerate(f, 1):
                            if rx.search(line):
                                hits.append(f"{fp}:{n}: {line.strip()[:160]}")
                                break
                except OSError:
                    continue
            if capped:
                break
    if not hits:
        return f"No files matched {pattern!r} (searched {scanned} text files)."
    out = "\n".join(hits)
    if capped:
        out += f"\n…[stopped at {max_results} files. Narrow the pattern or pass a `path`.]"
    return out


_TEXTY = {"txt", "md", "log", "csv", "tsv", "json", "yaml", "yml", "xml", "html", "htm",
          "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "go", "rs", "rb",
          "php", "sh", "sql", "ini", "cfg", "conf", "toml"}


def write_note(text: str) -> str:
    from harness import current_run_dir

    d = current_run_dir()
    with open(os.path.join(d, "notes.md"), "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    return f"Noted ({len(text)} chars). Notes persist across context compaction."


def write_deliverable(kind: str, filename: str, spec: str = "") -> str:
    kind = (kind or "").lower().strip()
    valid = ("xlsx", "docx", "pptx", "md")
    if kind not in valid:
        return f"Unknown kind {kind!r}. Valid kinds: {', '.join(valid)}."
    if not spec:
        return (
            f"`spec` is missing, so there is nothing to write. Send the complete content "
            f"in the same call.\nExample for {kind}: {_SPEC_EXAMPLES[kind]}"
        )
    if isinstance(spec, (dict, list)):
        payload = spec
    else:
        text = str(spec).strip()
        looks_json = text.startswith("{") or text.startswith("[")
        try:
            payload = json.loads(text, strict=False)
        except (TypeError, json.JSONDecodeError) as e:


            if kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(text, kind)
            else:
                return (
                    f"`spec` must be a JSON object, but it did not parse: {e}\n"
                    f"Received: {str(spec)[:200]}\n"
                    f"Example for {kind}: {_SPEC_EXAMPLES[kind]}"
                )
        else:

            if isinstance(payload, str) and kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(payload, kind)
            elif not looks_json and kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(text, kind)

    os.makedirs(OUTPUT_DIR, exist_ok=True)


    name = os.path.basename(filename or "").strip() or f"output.{kind}"
    if not name.lower().endswith(f".{kind}"):
        name = f"{os.path.splitext(name)[0]}.{kind}"
    path = os.path.join(OUTPUT_DIR, name)
    try:
        return deliver.write(kind, path, payload)
    except deliver.SpecError as e:
        return f"{e}\nExample for {kind}: {_SPEC_EXAMPLES[kind]}"
    except Exception as e:
        log.exception("deliverable failed")
        return f"Could not write {path}: {e}"


PROSE_KINDS = ("docx", "md", "pptx")

_SPEC_EXAMPLES = {
    "xlsx": '{"sheets":[{"name":"Data","columns":[{"header":"Region"},{"header":"Revenue","format":"currency"}],"rows":[["EMEA",1240000]]}]}',
    "docx": '{"title":"Report","blocks":[{"type":"heading","level":2,"text":"Findings"},{"type":"paragraph","text":"..."}]}',
    "pptx": '{"title":"Deck","slides":[{"layout":"bullets","title":"Findings","bullets":["a","b"]}]}',
    "md": '{"title":"Notes","blocks":[{"type":"paragraph","text":"..."}]}',
}


TOOL_FUNCTIONS = {
    "search_documents": search_documents,
    "read_chunk": read_chunk,
    "read_chunks": read_chunks,
    "list_folder": list_folder,
    "read_file": read_file,
    "grep_files": grep_files,
    "write_note": write_note,
    "write_deliverable": write_deliverable,
}

READ_ONLY = {"search_documents", "read_chunk", "read_chunks", "list_folder", "read_file", "grep_files"}


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": {"type": "object", "properties": props, "required": required}},
    }


TOOL_SCHEMAS = [
    _schema("search_documents",
            "Search indexed company documents. Returns SNIPPETS — call read_chunk or read_chunks for the "
            "full passage of results you want to quote or cite. Can optionally filter to a specific filename.",
            {"query": {"type": "string", "description": "what to look for, in plain words"},
             "k": {"type": "integer", "description": "how many results, 1-15", "default": 6},
             "filename": {"type": "string", "description": "optional filename to restrict search to a specific document or attachment"}},
            ["query"]),
    _schema("read_chunk",
            "Get the full text of one search result (or comma-separated ids), by the id shown in search_documents output. "
            "This is what you quote and cite from.",
            {"chunk_id": {"type": "string", "description": "an id from search_documents, e.g. a1b2_c0"}},
            ["chunk_id"]),
    _schema("read_chunks",
            "Get the full text of multiple search results in a single step. Highly recommended for inspecting "
            "multiple passages across documents or distant pages.",
            {"chunk_ids": {"type": "array", "items": {"type": "string"}, "description": "list of chunk IDs from search_documents"}},
            ["chunk_ids"]),
    _schema("list_folder",
            "List files and subfolders. Call with no path to see which root folders you may access.",
            {"path": {"type": "string", "description": "folder path; omit to list granted roots"},
             "pattern": {"type": "string", "description": "optional regex to filter names"}},
            []),
    _schema("read_file",
            "Read a file's raw text by line range. Prefer search_documents for indexed documents; "
            "use this for files the index does not cover.",
            {"path": {"type": "string"},
             "offset": {"type": "integer", "description": "first line, 0-based", "default": 0},
             "limit": {"type": "integer", "description": "how many lines, max 400", "default": 200}},
            ["path"]),
    _schema("grep_files",
            "Find which files contain a pattern. Returns file:line locations, not full context — "
            "follow up with read_file. Only searches text-like files.",
            {"pattern": {"type": "string", "description": "regex or plain substring"},
             "path": {"type": "string", "description": "optional folder to limit the search to"},
             "max_results": {"type": "integer", "default": 40}},
            ["pattern"]),
    _schema("write_note",
            "Save a finding to your scratchpad. Use for facts and citations you will need later — "
            "notes survive even when older conversation is compacted away.",
            {"text": {"type": "string"}}, ["text"]),
    _schema("write_deliverable",
            "Create a real file for the user and save it on this machine: Excel workbook, Word "
            "document, PowerPoint deck or Markdown. Write the full content in this one call.",
            {"kind": {"type": "string", "enum": ["xlsx", "docx", "pptx", "md"]},
             "filename": {"type": "string", "description": "e.g. q3_summary.xlsx"},


             "spec": {"type": "string",
                      "description": "the content. For docx/md/pptx just write plain Markdown "
                                     "(headings, paragraphs, - bullets, | tables |) — no JSON "
                                     'needed. For xlsx send JSON: {"sheets":[{"name":"Data",'
                                     '"columns":["Part","Qty"],"rows":[["BOLT-M12",40]]}]}'}},
            ["kind", "filename", "spec"]),
]

assert {s["function"]["name"] for s in TOOL_SCHEMAS} == set(TOOL_FUNCTIONS), "schema/function mismatch"


def demo() -> None:
    import glob
    import shutil

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


    out = write_deliverable("xlsx", "parts", json.dumps(
        {"columns": ["Part", "Qty"], "rows": [["BOLT-M12", 40], ["BOLT-M8", 25]]}))
    assert out.startswith("Wrote"), out
    written = glob.glob(os.path.join(OUTPUT_DIR, "*"))
    assert [os.path.basename(p) for p in written] == ["parts.xlsx"], written


    empty = write_deliverable("xlsx", "empty.xlsx", json.dumps({"columns": ["Part", "Qty"]}))
    assert "no rows" in empty and "would" in empty, empty
    assert not os.path.exists(os.path.join(OUTPUT_DIR, "empty.xlsx")), "refused spec must write nothing"


    md_out = write_deliverable("docx", "letter", "# Request\n\nDear Sir,\n\nI can't pay the \"fee\".")
    assert md_out.startswith("Wrote"), md_out
    assert os.path.exists(os.path.join(OUTPUT_DIR, "letter.docx"))
    assert "did not parse" in write_deliverable("xlsx", "y.xlsx", "# not json")
    assert "is missing" in write_deliverable("docx", "z.docx", "")

    assert "Unknown kind" in write_deliverable("pdf", "x.pdf", "{}")
    assert "did not parse" in write_deliverable("xlsx", "x.xlsx", "{not json")
    assert "Example for xlsx" in write_deliverable("xlsx", "x.xlsx", "{not json")

    assert "Access denied" in read_file("C:\\Windows\\System32\\drivers\\etc\\hosts")
    assert "Access denied" in list_folder("C:\\Windows")
    assert "No chunk with id" in read_chunk("bogus_c0")
    assert "Invalid regex" in grep_files("[unclosed")

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    print("PASS tools — extension fixup, empty-deliverable refusal, permission and arg errors")


if __name__ == "__main__":
    demo()
