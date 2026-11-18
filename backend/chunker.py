import re
try:
    from config import CHUNK_OVERLAP, CHUNK_SIZE
    TARGET_CHARS = CHUNK_SIZE
    OVERLAP_CHARS = CHUNK_OVERLAP
except Exception:
    TARGET_CHARS = 1000
    OVERLAP_CHARS = 150

MIN_CHUNK_CHARS = 60

_PARA = re.compile(r"\n\s*\n")
_SENT = re.compile(r"(?<=[.!?])\s+")


_ROWISH = re.compile(r"^\s*(\||[^:\n]{1,60}:\s)", re.M)


def _split_oversized(
    text: str,
    target_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[str]:
    target = target_chars or TARGET_CHARS
    overlap = overlap_chars if overlap_chars is not None else OVERLAP_CHARS
    if len(text) <= target:
        return [text]
    out, rest = [], text
    while len(rest) > target:
        window = rest[:target]
        cut = -1
        for m in _SENT.finditer(window):
            cut = m.end()
        if cut < MIN_CHUNK_CHARS:
            cut = window.rfind(" ")
        if cut < MIN_CHUNK_CHARS:
            cut = target
        out.append(rest[:cut].strip())
        rest = rest[max(0, cut - overlap):].lstrip()
    if rest.strip():
        out.append(rest.strip())
    return [c for c in out if c]


def chunk_page(
    text: str,
    kind: str = "text",
    target_chars: int | None = None,
    overlap_chars: int | None = None,
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []

    target = target_chars or TARGET_CHARS
    overlap = overlap_chars if overlap_chars is not None else OVERLAP_CHARS


    if kind in ("table", "sheet") or len(_ROWISH.findall(text)) > 3:
        return _split_on_lines(text)

    chunks, buf = [], ""
    for para in _PARA.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > target:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_oversized(para, target_chars=target, overlap_chars=overlap))
            continue
        if buf and len(buf) + len(para) + 2 > target:
            chunks.append(buf)
            buf = buf[-overlap:].lstrip() + "\n\n" + para if overlap else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf:
        chunks.append(buf)

    merged = _merge_runts(chunks)
    return [c for c in merged if len(c) >= MIN_CHUNK_CHARS or len(merged) == 1]


def _split_on_lines(text: str) -> list[str]:
    lines = text.split("\n")
    header = lines[0] if lines and len(lines[0]) < 200 else ""
    chunks, buf = [], []
    size = 0
    for line in lines:
        if size + len(line) > TARGET_CHARS and buf:
            chunks.append("\n".join(buf))
            buf = [header] if header else []
            size = len(header) + 1
        buf.append(line)
        size += len(line) + 1
    if buf:
        chunks.append("\n".join(buf))
    return [c.strip() for c in chunks if c.strip()]


def _merge_runts(chunks: list[str]) -> list[str]:
    out: list[str] = []
    for c in chunks:
        if out and len(c) < MIN_CHUNK_CHARS:
            out[-1] = f"{out[-1]}\n\n{c}"
        else:
            out.append(c)
    return out


def context_line(filename: str, section: str, page: int) -> str:
    parts = [filename]
    if section:
        parts.append(section)
    parts.append(f"page {page}")
    return " > ".join(parts)


def embed_input(filename: str, section: str, page: int, text: str) -> str:
    return f"{context_line(filename, section, page)}\n{text}"


def demo() -> None:
    prose = "\n\n".join(f"Paragraph {i}. " + "The pump operates normally. " * 12 for i in range(6))
    cs = chunk_page(prose)
    assert len(cs) > 1, "long prose should split"
    assert all(len(c) <= TARGET_CHARS + OVERLAP_CHARS + 200 for c in cs), [len(c) for c in cs]
    assert all(len(c) >= MIN_CHUNK_CHARS for c in cs), "no runt chunks"

    short = chunk_page("Just one line about pump P-101.")
    assert short == ["Just one line about pump P-101."], short

    assert chunk_page("") == [] and chunk_page("   ") == []

    table = "Region | Revenue | Qty\n" + "\n".join(f"R{i} | {i*100} | {i}" for i in range(120))
    tc = chunk_page(table, kind="table")
    assert len(tc) > 1, "big table should split"
    assert all(c.startswith("Region | Revenue | Qty") for c in tc), "header repeated in every chunk"
    for c in tc:
        for line in c.split("\n"):
            assert line.count("|") in (0, 2), f"row split across chunks: {line!r}"

    rows = "\n".join(f"name: item{i}\nqty: {i}\n" for i in range(200))
    assert len(chunk_page(rows, kind="sheet")) > 1

    nospace = "x" * 5000
    assert len(chunk_page(nospace)) > 1, "must not loop forever with no boundaries"

    assert embed_input("m.pdf", "Ch 3 > Safety", 7, "body").startswith("m.pdf > Ch 3 > Safety > page 7\n")
    assert context_line("m.pdf", "", 2) == "m.pdf > page 2"

    print(f"PASS chunker — prose={len(cs)} table={len(tc)} chunks")


if __name__ == "__main__":
    demo()
