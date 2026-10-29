"""The agent's tools.

Three rules shape every function here, all of them consequences of driving an 8B local
model rather than a frontier one:

  1. **A tool never raises.** It returns a string telling the model what to do next. An
     exception ends the turn and the user sees nothing; a good error message is a free
     retry. Every failure message contains the schema or the valid options, so the model
     only has to copy, not remember.

  2. **Every output has a hard cap, and truncation is announced with the true size.** A
     model that knows it saw 200 of 1,847 lines will paginate. A model that thinks it saw
     the whole file will confidently make up the rest.

  3. **Eight tools, and reads are cheap two-hop.** Tool-selection accuracy falls off
     sharply past ~10 tools for small models. search_documents returns snippets, not full
     passages — reconnaissance first, then read_chunk for the two or three that matter.

Path access is checked against the granted-folder allow-list on every call, including the
read-only ones. Document contents are untrusted data: a PDF that says "ignore previous
instructions" is the realistic attack here, so retrieved text is always delimited and
labelled as data in the prompt.
"""
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


# ---------------------------------------------------------------------------
# Retrieval — two hops
# ---------------------------------------------------------------------------


def search_documents(query: str, k: int = 6) -> str:
    """Hop one: cheap reconnaissance over the index."""
    k = max(1, min(int(k or 6), 10))
    hits = search(query, k=k)
    if not hits:
        stats = store.queue_stats()
        if stats["indexed"] == 0:
            return (
                "No documents are indexed yet. "
                f"{stats['pending']} files are still queued. "
                "Tell the user indexing is still in progress rather than guessing an answer."
            )
        return (
            f"No matches for {query!r} across {stats['indexed']} indexed files.\n"
            "Try different wording, or call list_folder to see what exists. "
            "If the corpus genuinely does not cover this, say so — do not guess."
        )

    conf = confidence(hits)
    lines = [f"Retrieval confidence: {conf}. {len(hits)} results (snippets — call read_chunk for full text):"]
    for h in hits:
        snippet = " ".join(h["text"].split())[:MAX_SNIPPET]
        both = "both" if h["agreed"] else ("keyword" if h["dense_rank"] is None else "semantic")
        lines.append(f"- id={h['id']} [{h['filename']}:{h['page']}] ({both}) {snippet}…")
    if conf in ("low", "none"):
        lines.append(
            "Confidence is low: nothing matched strongly. Prefer telling the user you "
            "don't have enough information over answering from these."
        )
    return "\n".join(lines)


def read_chunk(chunk_id: str) -> str:
    """Hop two: the full passage behind one search result."""
    rec = store.get_chunks([chunk_id]).get(chunk_id)
    if not rec:
        return (
            f"No chunk with id {chunk_id!r}. Ids come from search_documents results and "
            "look like 'a1b2c3d4_p3_c0'. Call search_documents first."
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


# ---------------------------------------------------------------------------
# Filesystem traversal
# ---------------------------------------------------------------------------


def list_folder(path: str = "", pattern: str = "") -> str:
    """Folder listing. Filenames, sizes and folder structure are relevance signal the
    model gets for free — 'Q3_Budget_FINAL.xlsx' in '\\Finance\\2025\\Approved' says more
    per token than 500 characters of its contents."""
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
    """Raw file read by line range. For indexed documents prefer search_documents +
    read_chunk; this is for files the index doesn't cover, or for checking something
    exactly as written."""
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
    """Literal/regex search across granted folders. Cheap reconnaissance: returns where
    matches are, not the surrounding context, so the first hop stays small."""
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
                                break  # one hit per file keeps reconnaissance cheap
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


# ---------------------------------------------------------------------------
# Working state and deliverables
# ---------------------------------------------------------------------------


def write_note(text: str) -> str:
    """The agent's scratchpad. Findings written here survive context compaction, because
    they live on disk rather than in the transcript."""
    from harness import current_run_dir

    d = current_run_dir()
    with open(os.path.join(d, "notes.md"), "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n\n")
    return f"Noted ({len(text)} chars). Notes persist across context compaction."


def write_deliverable(kind: str, filename: str, spec: str = "") -> str:
    """Produce a real file. `spec` is JSON; see deliver.py for each kind's shape.

    Deliberately one coarse tool rather than create/add_heading/add_row/save: a small
    model gets a five-call stateful sequence wrong often enough that it regularly leaves
    a half-written, corrupt file behind. One validated call either works or fails
    cleanly with a message it can act on.
    """
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
            # Prose formats accept Markdown directly. Requiring JSON here means every
            # apostrophe, quote and newline in a letter is a chance to emit invalid JSON,
            # and a complete, well-written application was being discarded over a single
            # unescaped quote. Markdown is what these models are best at, so take it.
            if kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(text, kind)
            else:
                return (
                    f"`spec` must be a JSON object, but it did not parse: {e}\n"
                    f"Received: {str(spec)[:200]}\n"
                    f"Example for {kind}: {_SPEC_EXAMPLES[kind]}"
                )
        else:
            # Valid JSON that is merely a quoted string is still prose, not a spec.
            if isinstance(payload, str) and kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(payload, kind)
            elif not looks_json and kind in PROSE_KINDS:
                payload = deliver.spec_from_markdown(text, kind)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Models routinely pass a bare stem ("parts"), which would write an extensionless file
    # Windows can't open in Excel. The kind already says what it is, so fix it silently.
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


# Shown verbatim in every failure message, so each is a complete, valid spec the model
# can copy and adapt rather than a schema it has to interpret.
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
    "list_folder": list_folder,
    "read_file": read_file,
    "grep_files": grep_files,
    "write_note": write_note,
    "write_deliverable": write_deliverable,
}

READ_ONLY = {"search_documents", "read_chunk", "list_folder", "read_file", "grep_files"}


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {"name": name, "description": desc,
                     "parameters": {"type": "object", "properties": props, "required": required}},
    }


TOOL_SCHEMAS = [
    _schema("search_documents",
            "Search indexed company documents. Returns SNIPPETS only — call read_chunk for the "
            "full passage of any result you want to quote or cite. Do NOT use this to list files; "
            "use list_folder.",
            {"query": {"type": "string", "description": "what to look for, in plain words"},
             "k": {"type": "integer", "description": "how many results, 1-10", "default": 6}},
            ["query"]),
    _schema("read_chunk",
            "Get the full text of one search result, by the id shown in search_documents output. "
            "This is what you quote and cite from.",
            {"chunk_id": {"type": "string", "description": "an id from search_documents, e.g. a1b2_p3_c0"}},
            ["chunk_id"]),
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
             # A STRING, deliberately, even though the value is JSON. Declaring this as
             # {"type": "object"} is avoided because some tool-call parsers drop any call
             # whose arguments contain a nested object, returning an empty message with no
             # content and no tool_calls.
             # Do not change this back without validation. The parsing this costs us is handled in deliver.py, which
             # accepts every spec shape a model plausibly emits.
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

    # A bare stem must still produce a file Excel can open.
    out = write_deliverable("xlsx", "parts", json.dumps(
        {"columns": ["Part", "Qty"], "rows": [["BOLT-M12", 40], ["BOLT-M8", 25]]}))
    assert out.startswith("Wrote"), out
    written = glob.glob(os.path.join(OUTPUT_DIR, "*"))
    assert [os.path.basename(p) for p in written] == ["parts.xlsx"], written

    # Headers with no data must be refused, not written as an empty file the model then
    # claims is full — this was a real failure observed end to end.
    empty = write_deliverable("xlsx", "empty.xlsx", json.dumps({"columns": ["Part", "Qty"]}))
    assert "no rows" in empty and "would" in empty, empty
    assert not os.path.exists(os.path.join(OUTPUT_DIR, "empty.xlsx")), "refused spec must write nothing"

    # Prose kinds take Markdown directly; xlsx still requires JSON, because a sheet is
    # real structured data and Markdown would lose the column types.
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
