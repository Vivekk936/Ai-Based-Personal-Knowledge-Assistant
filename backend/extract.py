"""Format-aware text extraction: turns any supported file into a stream of small,
individually-citable Pages. Splitting matters more than parsing — a whole 60-sheet
workbook or a 200-page manual as one blob retrieves as one undifferentiated hit, so
every format is broken at its own natural seam (PDF page, docx heading, sheet row
group, slide) and the seam name is kept in `section` so a citation can say where it
came from. Nothing here rasterizes, OCRs or calls a model: pages with no text layer
are flagged `needs_ocr` and handed back for the caller to decide about.
"""
import csv
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from html.parser import HTMLParser

log = logging.getLogger("ada.extract")

MIN_PDF_TEXT_CHARS = 40  # below this a PDF page is a scan, not a text layer
CHUNK_CHARS = 8000
ROWS_PER_PAGE = 40
# A spreadsheet is the one format where a single file can hold a million rows of
# machine-generated data that no one will ever ask a question about. Read the first
# 20k and say so, rather than emitting 25,000 pages for one file.
MAX_SHEET_ROWS = 20_000

TEXT_EXTENSIONS = {"txt", "md", "log", "json", "yaml", "yml", "xml", "html", "htm"}
CODE_EXTENSIONS = {"py", "js", "ts", "tsx", "jsx", "java", "c", "cpp", "h", "go", "rs", "rb", "php", "sh", "sql"}
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp", "gif"}
SUPPORTED_EXTENSIONS = (
    {"pdf", "docx", "xlsx", "xlsm", "pptx", "csv", "tsv"} | TEXT_EXTENSIONS | CODE_EXTENSIONS | IMAGE_EXTENSIONS
)

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
_CODE_DEF = re.compile(r"^(?:def|class|function|func|public|private)\b", re.M)
_BLANK_RUN = re.compile(r"\n{4,}")
_BLOCK_TAGS = {"p", "div", "br", "li", "tr", "td", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}


@dataclass
class Page:
    number: int
    text: str
    kind: str
    section: str = ""
    needs_ocr: bool = False


def extension_of(path: str) -> str:
    return os.path.splitext(path)[1].lower().lstrip(".")


def is_supported(path: str) -> bool:
    return extension_of(path) in SUPPORTED_EXTENSIONS


def extract(path: str, *, max_pages: int = 5000) -> Iterator[Page]:
    """Lazy page stream. A corrupt or unreadable file yields nothing (logged); only a
    genuinely unsupported extension raises — and it raises here, at call time, rather
    than on the first next() like it would from a plain generator function."""
    ext = extension_of(path)
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type: .{ext}")
    return _stream(path, ext, max_pages)


def _stream(path: str, ext: str, max_pages: int) -> Iterator[Page]:
    if ext == "pdf":
        pages = _pdf(path)
    elif ext == "docx":
        pages = _docx(path)
    elif ext in ("xlsx", "xlsm"):
        pages = _xlsx(path)
    elif ext == "pptx":
        pages = _pptx(path)
    elif ext in ("csv", "tsv"):
        pages = _delimited(path, ext)
    elif ext in IMAGE_EXTENSIONS:
        pages = _image(path)
    else:
        pages = _plain(path, ext)

    emitted = 0
    try:
        for page in pages:
            page.text = _clean(page.text)
            if not page.text and not page.needs_ocr:
                continue
            yield page
            emitted += 1
            if emitted >= max_pages:
                return
    except Exception as e:  # unreadable/corrupt is a normal outcome in a real folder
        log.warning("extract failed for %s: %s", path, e)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # One blank line separates paragraphs and two mark a section break — both carry
    # meaning, so only longer runs (page-break gutters, empty table cells) collapse.
    return _BLANK_RUN.sub("\n\n\n", text).strip()


def _split_chars(text: str) -> list[str]:
    return [text[i : i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]


def _table_blocks(rows: list[list[str]]) -> Iterator[str]:
    """Pipe-delimited, header repeated on every block — a retrieval hit on row 300 of
    a long table is useless if the column names stayed behind on the first block."""
    if not rows:
        return
    header = " | ".join(rows[0])
    if len(rows) == 1:
        yield header
        return
    for i in range(1, len(rows), ROWS_PER_PAGE):
        yield "\n".join([header] + [" | ".join(r) for r in rows[i : i + ROWS_PER_PAGE]])


def _row_pages(number: int, section: str, header: list[str], rows, note: str) -> Iterator[Page]:
    """Schema page first, then row groups rendered as `header: value` — so a hit lands
    on the handful of rows that actually answer the question, not the whole sheet."""
    yield Page(number, f"{section}\nColumns ({len(header)}): {' | '.join(header)}\n{note}", "sheet", section)

    group, taken = [], 0
    for row in rows:
        if taken >= MAX_SHEET_ROWS:
            break
        cells = ["" if v is None else str(v).strip() for v in row]
        if not any(cells):
            continue
        taken += 1
        group.append(
            " | ".join(f"{header[i] if i < len(header) else f'col{i + 1}'}: {v}" for i, v in enumerate(cells) if v)
        )
        if len(group) >= ROWS_PER_PAGE:
            yield Page(number, "\n".join(group), "table", section)
            group = []
    if group:
        yield Page(number, "\n".join(group), "table", section)


# Third-party parsers are imported per format rather than at module level: importing
# openpyxl + python-pptx + Pillow + pdfium on every server start costs ~a second for
# libraries most requests never touch.


def _pdf(path: str) -> Iterator[Page]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(path)
    try:
        title = (pdf.get_metadata_value("Title") or "").strip()
        marks = _pdf_outline(pdf)
        heading = ""
        for i in range(len(pdf)):
            heading = marks.get(i, heading)
            section = " > ".join(p for p in (title, heading) if p)
            try:
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_range()
                textpage.close()
                page.close()
            except Exception as e:  # one broken page must not cost the other 199
                log.warning("%s page %d unreadable: %s", path, i + 1, e)
                continue
            if len(text.strip()) < MIN_PDF_TEXT_CHARS:
                yield Page(i + 1, "", "scanned", section, needs_ocr=True)
            else:
                yield Page(i + 1, text, "text", section)
    finally:
        pdf.close()


def _pdf_outline(pdf) -> dict[int, str]:
    """Bookmark title by 0-based page index. Flat, not a nested path: pdfium doesn't
    hand back bookmark depth, and a nearest-heading label is most of the value."""
    marks = {}
    try:
        for bookmark in pdf.get_toc():
            dest = bookmark.get_dest()
            if dest:
                marks.setdefault(dest.get_index(), bookmark.get_title())
    except Exception as e:
        log.debug("no usable PDF outline: %s", e)
    return marks


def _docx(path: str) -> Iterator[Page]:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = docx.Document(path)
    heads: list[str] = []
    buf: list[str] = []
    n = 0

    # Body order, not paragraphs-then-tables: a table belongs where it was written,
    # under the heading it was written under.
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            para = Paragraph(child, document)
            style = getattr(para.style, "name", "") or ""
            if style.startswith("Heading") and para.text.strip():
                if buf:
                    n += 1
                    yield Page(n, "\n".join(buf), "text", " > ".join(heads))
                    buf = []
                depth = re.search(r"\d", style)
                level = int(depth.group()) if depth else 1
                heads = heads[: level - 1] + [para.text.strip()]
            elif para.text.strip():
                buf.append(para.text)
        elif tag == "tbl":
            if buf:
                n += 1
                yield Page(n, "\n".join(buf), "text", " > ".join(heads))
                buf = []
            try:
                rows = [[c.text.strip() for c in r.cells] for r in Table(child, document).rows]
            except Exception as e:
                log.warning("%s: skipping unreadable table: %s", path, e)
                continue
            for block in _table_blocks(rows):
                n += 1
                yield Page(n, block, "table", " > ".join(heads))

    if buf:
        n += 1
        yield Page(n, "\n".join(buf), "text", " > ".join(heads))


def _xlsx(path: str) -> Iterator[Page]:
    import openpyxl

    # read_only keeps a 200MB workbook streaming instead of building a cell object
    # per cell; data_only gives cached formula results instead of "=SUM(A1:A9)".
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        for index, ws in enumerate(wb.worksheets, start=1):
            section = f"Sheet: {ws.title}"
            try:
                rows = ws.iter_rows(values_only=True)
                first = next(rows, None)
                if first is None:
                    continue
                header = [str(v).strip() if v is not None else f"col{i}" for i, v in enumerate(first, start=1)]
                declared = ws.max_row or 0
                note = f"Rows: {declared}, Columns: {ws.max_column or len(header)}"
                if declared > MAX_SHEET_ROWS:
                    note += f" (truncated to the first {MAX_SHEET_ROWS} rows)"
                yield from _row_pages(index, section, header, rows, note)
            except Exception as e:  # one corrupt sheet must not lose the workbook
                log.warning("%s sheet %r unreadable: %s", path, ws.title, e)
    finally:
        wb.close()


def _pptx(path: str) -> Iterator[Page]:
    from pptx import Presentation

    for index, slide in enumerate(Presentation(path).slides, start=1):
        try:
            title_shape = slide.shapes.title
            title = title_shape.text.strip() if title_shape is not None else ""
            parts = []
            for shape in slide.shapes:
                if shape.has_table:
                    rows = [[c.text.strip() for c in r.cells] for r in shape.table.rows]
                    parts.extend(_table_blocks(rows))
                elif shape.has_text_frame and shape.text_frame.text.strip():
                    parts.append(shape.text_frame.text)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text.strip():
                parts.append("Notes: " + slide.notes_slide.notes_text_frame.text)
            yield Page(index, "\n".join(parts), "slide", f"Slide {index}: {title}")
        except Exception as e:
            log.warning("%s slide %d unreadable: %s", path, index, e)


def _delimited(path: str, ext: str) -> Iterator[Page]:
    # utf-8-sig drops Excel's BOM; errors="replace" means a latin-1 export degrades to
    # a few odd characters instead of an exception.
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = "excel-tab" if ext == "tsv" else "excel"
        reader = csv.reader(f, dialect)
        first = next(reader, None)
        if first is None:
            return
        header = [(v or f"col{i}").strip() for i, v in enumerate(first, start=1)]
        section = f"Sheet: {os.path.basename(path)}"
        yield from _row_pages(1, section, header, reader, f"Columns: {len(header)}")


def _image(path: str) -> Iterator[Page]:
    from PIL import Image

    with Image.open(path) as im:
        section = f"Image {im.width}x{im.height} {im.mode}"
    yield Page(1, "", "image", section, needs_ocr=True)


def _plain(path: str, ext: str) -> Iterator[Page]:
    if ext in ("md", "html", "htm") or ext in CODE_EXTENSIONS:
        # These are split by structure, which needs the whole text — bounded in
        # practice because the caller size-caps files before ingesting them.
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        blocks = _markdown_blocks(text) if ext == "md" else _code_blocks(text) if ext in CODE_EXTENSIONS else [("", _strip_html(text))]
        n = 0
        for section, body in blocks:
            for chunk in _split_chars(body):
                n += 1
                yield Page(n, chunk, "text", section)
        return

    # Plain text is the one format that is routinely gigabytes (logs, dumps), so it
    # is read a page at a time and never lands in memory whole.
    with open(path, encoding="utf-8", errors="replace") as f:
        n = 0
        while True:
            block = f.read(CHUNK_CHARS)
            if not block:
                return
            n += 1
            yield Page(n, block, "text", "")


def _markdown_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(_MD_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    blocks, heads = [], []
    if text[: matches[0].start()].strip():
        blocks.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heads = heads[: len(m.group(1)) - 1] + [m.group(2).strip()]
        blocks.append((" > ".join(heads), text[m.start() : end]))
    return blocks


def _code_blocks(text: str) -> list[tuple[str, str]]:
    starts = [m.start() for m in _CODE_DEF.finditer(text)]
    if len(starts) < 2:
        return [("", text)]
    blocks = [("", text[: starts[0]])] if text[: starts[0]].strip() else []
    for start, end in zip(starts, starts[1:] + [len(text)]):
        body = text[start:end]
        blocks.append((body.split("\n", 1)[0].strip()[:80], body))
    return blocks


class _TextOnly(HTMLParser):
    """Minimal tag stripper — bs4/lxml would be a dependency for twenty lines."""

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self.skip = max(0, self.skip - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def _strip_html(text: str) -> str:
    parser = _TextOnly()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts)


def demo() -> None:
    """Builds one real file per family, extracts it, and asserts the splitting
    actually happened. Run: python extract.py"""
    import ctypes
    import tempfile

    import openpyxl
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c
    from docx import Document
    from PIL import Image
    from pptx import Presentation
    from pptx.util import Inches

    tmp = tempfile.mkdtemp(prefix="ada_extract_")
    out = lambda name: os.path.join(tmp, name)  # noqa: E731

    # --- docx: headings + an interleaved table, non-ASCII filename on purpose ---
    docx_path = out("Bericht Grüße 文書.docx")
    d = Document()
    d.add_heading("Chapter 3", level=1)
    d.add_paragraph("Intro paragraph about the plant.")
    d.add_heading("Safety", level=2)
    d.add_paragraph("Always lock out the breaker before service.")
    table = d.add_table(rows=3, cols=2)
    for r, (a, b) in enumerate([("Component", "Limit"), ("Pump P-101", "150 PSI"), ("Valve V-9", "80 PSI")]):
        table.rows[r].cells[0].text, table.rows[r].cells[1].text = a, b
    d.add_heading("Maintenance", level=2)
    d.add_paragraph("Grease the bearings quarterly.")
    d.save(docx_path)

    pages = list(extract(docx_path))
    sections = [p.section for p in pages]
    assert len(pages) >= 4, f"docx should split on headings, got {len(pages)} pages"
    assert any(s == "Chapter 3 > Safety" for s in sections), sections
    assert any(s == "Chapter 3 > Maintenance" for s in sections), sections
    table_pages = [p for p in pages if p.kind == "table"]
    assert table_pages and "Component | Limit" in table_pages[0].text
    assert "Pump P-101 | 150 PSI" in table_pages[0].text
    # table sits under the heading it was written under, i.e. body order was preserved
    assert table_pages[0].section == "Chapter 3 > Safety", table_pages[0].section
    print(f"PASS docx    {len(pages)} pages, {len(table_pages)} table page(s), sections {sorted(set(sections))}")

    # --- xlsx: 100 rows on sheet 1 must not come back as one page ---
    xlsx_path = out("book.xlsx")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Q3"
    ws.append(["Region", "Revenue"])
    for i in range(100):
        ws.append([f"Region {i}", 1000 + i])
    ws2 = wb.create_sheet("Notes")
    ws2.append(["Note"])
    ws2.append(["Renewal due in April"])
    wb.save(xlsx_path)

    pages = list(extract(xlsx_path))
    sheet1 = [p for p in pages if p.number == 1]
    assert pages[0].kind == "sheet" and pages[0].section == "Sheet: Q3", pages[0]
    assert "Region | Revenue" in pages[0].text
    assert len(sheet1) > 2, f"100 rows should split into row groups, got {len(sheet1)} pages"
    assert any("Region: Region 42 | Revenue: 1042" in p.text for p in sheet1)
    assert any(p.section == "Sheet: Notes" and "Renewal due in April" in p.text for p in pages)
    print(f"PASS xlsx    {len(pages)} pages ({len(sheet1)} for the 100-row sheet), 2 sheets seen")

    # --- pptx ---
    pptx_path = out("deck.pptx")
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = "Agenda"
    s1.placeholders[1].text = "Budget review\nHiring plan"
    s1.notes_slide.notes_text_frame.text = "Keep this under ten minutes."
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    s2.shapes.title.text = "Numbers"
    tbl = s2.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    tbl.cell(0, 0).text, tbl.cell(0, 1).text = "Quarter", "Total"
    tbl.cell(1, 0).text, tbl.cell(1, 1).text = "Q3", "1.2M"
    prs.save(pptx_path)

    pages = list(extract(pptx_path))
    assert len(pages) == 2 and all(p.kind == "slide" for p in pages), pages
    assert pages[0].section == "Slide 1: Agenda", pages[0].section
    assert "Hiring plan" in pages[0].text and "Keep this under ten minutes." in pages[0].text
    assert "Quarter | Total" in pages[1].text and "Q3 | 1.2M" in pages[1].text
    print(f"PASS pptx    {len(pages)} slides, notes and table content present")

    # --- csv ---
    csv_path = out("rows.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write("name,city\n")
        for i in range(90):
            f.write(f"Person {i},Køge\n")
    pages = list(extract(csv_path))
    assert pages[0].kind == "sheet" and "name | city" in pages[0].text
    assert len(pages) > 2, f"90 rows should split, got {len(pages)}"
    assert any("name: Person 42 | city: Køge" in p.text for p in pages)
    print(f"PASS csv     {len(pages)} pages (1 schema + row groups)")

    # --- md / txt / html ---
    md_path = out("notes.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Handbook\n\nWelcome.\n\n## Leave policy\n\nTwenty days a year.\n\n## Expenses\n\nSubmit monthly.\n")
    pages = list(extract(md_path))
    assert [p.section for p in pages] == ["Handbook", "Handbook > Leave policy", "Handbook > Expenses"], pages
    assert "Twenty days a year." in pages[1].text
    print(f"PASS md      {len(pages)} sections: {[p.section for p in pages]}")

    txt_path = out("big.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("line of text\n" * 2000)
    pages = list(extract(txt_path))
    assert len(pages) > 1, "a 26KB text file should chunk"
    assert all(len(p.text) <= CHUNK_CHARS for p in pages)
    print(f"PASS txt     {len(pages)} chunked pages")

    html_path = out("page.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("<html><head><style>p{color:red}</style></head><body><h1>Title</h1><p>Body &amp; more.</p></body></html>")
    pages = list(extract(html_path))
    assert len(pages) == 1 and "Body & more." in pages[0].text
    assert "color:red" not in pages[0].text, "style contents must be stripped"
    print("PASS html    tags stripped, script/style dropped")

    py_path = out("mod.py")
    with open(py_path, "w", encoding="utf-8") as f:
        f.write("import os\n\n\ndef alpha():\n    return 1\n\n\ndef beta():\n    return 2\n")
    pages = list(extract(py_path))
    assert len(pages) == 3, [p.section for p in pages]
    assert pages[1].section == "def alpha():" and "return 1" in pages[1].text
    print(f"PASS code    split into {len(pages)} blocks: {[p.section for p in pages]}")

    # --- image ---
    png_path = out("shot.png")
    Image.new("RGB", (320, 200), "white").save(png_path)
    pages = list(extract(png_path))
    assert len(pages) == 1 and pages[0].kind == "image"
    assert pages[0].needs_ocr and pages[0].text == ""
    assert pages[0].section == "Image 320x200 RGB", pages[0].section
    print(f"PASS image   1 page, needs_ocr=True, section={pages[0].section!r}")

    # --- pdf: page 1 has a text layer, page 2 is blank and must come back as scanned ---
    pdf_path = out("manual.pdf")
    doc = pdfium.PdfDocument.new()
    font = pdfium_c.FPDFText_LoadStandardFont(doc, b"Helvetica")
    for line in ["Pump P-101 maximum operating pressure is 150 PSI at the discharge flange.", None]:
        page = doc.new_page(595, 842)
        if line:
            obj = pdfium_c.FPDFPageObj_CreateTextObj(doc, font, 12)
            buf = ctypes.create_string_buffer(line.encode("utf-16-le") + b"\x00\x00")
            pdfium_c.FPDFText_SetText(obj, ctypes.cast(buf, pdfium_c.FPDF_WIDESTRING))
            pdfium_c.FPDFPageObj_Transform(obj, 1, 0, 0, 1, 50, 700)
            pdfium_c.FPDFPage_InsertObject(page, obj)
        pdfium_c.FPDFPage_GenerateContent(page)
    doc.save(pdf_path)
    doc.close()

    pages = list(extract(pdf_path))
    assert len(pages) == 2, pages
    assert pages[0].kind == "text" and "150 PSI" in pages[0].text and pages[0].number == 1
    assert pages[1].kind == "scanned" and pages[1].needs_ocr and pages[1].text == ""
    print("PASS pdf     2 pages, text layer read, blank page flagged needs_ocr")

    # --- failure modes: corrupt and missing must be silent zero-page outcomes ---
    for name, blob in [("broken.pdf", b"%PDF-1.4 not really"), ("broken.docx", b"PK\x03\x04junk"), ("broken.xlsx", b"nope")]:
        with open(out(name), "wb") as f:
            f.write(blob)
        assert list(extract(out(name))) == [], f"{name} should yield nothing"
    assert list(extract(out("does_not_exist.pdf"))) == []
    assert list(extract(out("does_not_exist.csv"))) == []
    print("PASS corrupt corrupt + missing files yield zero pages without raising")

    try:
        list(extract(out("archive.zip")))
        raise AssertionError("unsupported extension must raise ValueError")
    except ValueError:
        pass
    assert is_supported("C:\\x\\Grüße.DOCX") and not is_supported("x.zip")
    assert extension_of("C:\\x\\a.b.PDF") == "pdf"

    # max_pages is a hard stop, not a suggestion
    assert len(list(extract(csv_path, max_pages=2))) == 2
    print("PASS api     ValueError on unsupported ext, max_pages honored, helpers correct")
    print(f"\nAll checks passed. Scratch files in {tmp}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    demo()
