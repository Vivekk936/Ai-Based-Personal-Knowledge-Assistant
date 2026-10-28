"""Rule-based, zero-model organization suggestions: categorize loose files by
extension, flag exact duplicates (size match, then a cheap partial hash), flag
stale files by age. Pure metadata + hashing only within same-size groups — no
embeddings, no LLM calls. Organizing doesn't need understanding, just honest
heuristics, and that's what keeps it instant and free to run.
"""
import hashlib
import logging
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

from config import DATA_DIR
from store import list_files, SKIP_DIRS
from permissions import is_allowed

log = logging.getLogger("ada.organize")

STALE_DAYS = 365
MAX_FILES_FOR_DEDUP = 3000
MAX_SUGGESTIONS = 500

# Comprehensive categories covering PDFs, documents, spreadsheets, media, code,
# archives, executables, and miscellaneous ("other") files.
CATEGORIES = {
    "PDFs": {"pdf"},
    "Documents": {"doc", "docx", "txt", "md", "rtf", "odt", "epub", "pages", "tex", "wpd", "log"},
    "Spreadsheets": {"xls", "xlsx", "csv", "tsv", "ods", "xlsm", "numbers"},
    "Presentations": {"ppt", "pptx", "key", "odp", "pps"},
    "Images": {"jpg", "jpeg", "png", "gif", "bmp", "svg", "webp", "heic", "tiff", "ico", "psd", "ai", "raw"},
    "Videos": {"mp4", "mov", "avi", "mkv", "webm", "wmv", "flv", "m4v", "3gp"},
    "Audio": {"mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "opus", "aiff"},
    "Archives": {"zip", "rar", "7z", "tar", "gz", "bz2", "xz", "iso", "tgz", "7zip"},
    "Code": {
        "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp", "go", "rs",
        "html", "css", "json", "yaml", "yml", "sql", "sh", "bat", "ps1", "rb", "php",
        "dart", "xml", "toml", "ini", "cfg", "env"
    },
    "Installers": {"exe", "msi", "dmg", "pkg", "deb", "rpm", "apk", "appimage", "bin"},
}


def _category(ext: str) -> str:
    """Matches any file extension to a logical category, returning 'Other' for anything else."""
    ext_clean = ext.lower().lstrip(".")
    for name, exts in CATEGORIES.items():
        if ext_clean in exts:
            return name
    return "Other"


def _partial_hash(path: str, chunk: int = 65536) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(chunk))
    except OSError:
        return ""
    return h.hexdigest()


def suggest(root: str) -> dict:
    """Inspects a granted directory and suggests non-destructive moves for all file types
    (including PDFs, Office documents, media, code, and other loose files)."""
    if not is_allowed(root):
        raise PermissionError(f"{root} was not granted to Ada")

    root = os.path.abspath(root)
    files = list_files(root)

    # Fallback to direct directory scan if database has not catalogued this folder yet
    if not files:
        files = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS]
            for name in filenames:
                p = os.path.join(dirpath, name)
                try:
                    st = os.stat(p)
                    ext = os.path.splitext(name)[1].lower().lstrip(".") or "noext"
                    files.append({"path": p, "size": st.st_size, "mtime": st.st_mtime, "ext": ext})
                except OSError:
                    continue

    now = time.time()

    # Duplicate detection within same-size groups
    duplicate_paths: set[str] = set()
    if len(files) <= MAX_FILES_FOR_DEDUP:
        by_size = defaultdict(list)
        for f in files:
            by_size[f["size"]].append(f)
        for size, group in by_size.items():
            if size == 0 or len(group) < 2:
                continue
            by_hash = defaultdict(list)
            for f in group:
                by_hash[_partial_hash(f["path"])].append(f["path"])
            for paths in by_hash.values():
                if len(paths) > 1:
                    duplicate_paths.update(paths[1:])  # flag redundant copies

    rows = []
    category_dirs = set(CATEGORIES.keys()) | {"Other"}

    for f in files:
        path = f["path"]
        category = _category(f["ext"])
        age_days = (now - f["mtime"]) / 86400
        reasons = []
        dest = path

        rel_parent = os.path.relpath(os.path.dirname(path), root)

        if path in duplicate_paths:
            reasons.append("likely duplicate (review before deleting manually)")
        elif rel_parent == "." or rel_parent not in category_dirs:
            # File is loose in the root directory or in an unsorted directory
            dest = os.path.join(root, category, os.path.basename(path))
            if dest != path:
                reasons.append(f"categorize -> {category}")

        if age_days > STALE_DAYS:
            reasons.append(f"not modified in {int(age_days)} days")

        if reasons and dest != path:
            rows.append({
                "path": path,
                "suggested_destination": dest,
                "category": category,
                "reason": "; ".join(reasons)
            })

    return {
        "items": rows[:MAX_SUGGESTIONS],
        "total_candidates": len(rows),
        "truncated": len(rows) > MAX_SUGGESTIONS,
        "dedup_skipped": len(files) > MAX_FILES_FOR_DEDUP,
    }


def apply_moves(rows: list[dict]) -> dict:
    """Applies only rows where suggested_destination differs from path. Never touches
    duplicates (those stay flag-only), never overwrites an existing file, and re-checks
    each source still exists right before moving it."""
    log_path = DATA_DIR / "applied_moves.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    applied, skipped, errors = 0, 0, []

    with open(log_path, "a", encoding="utf-8") as move_log:
        for row in rows:
            src, dest = row["path"], row["suggested_destination"]
            if src == dest:
                continue
            if not is_allowed(src) or not is_allowed(dest):
                skipped += 1
                continue
            if not os.path.exists(src):
                skipped += 1
                log.warning("skip apply: source vanished since suggest: %s", src)
                continue
            if os.path.exists(dest):
                skipped += 1
                continue
            try:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(src, dest)
            except OSError as e:
                skipped += 1
                errors.append(f"{src}: {e}")
                log.error("failed to move %s -> %s: %s", src, dest, e)
                continue
            move_log.write(f"{time.time()},{src},{dest}\n")
            applied += 1

    return {"applied": applied, "skipped": skipped, "errors": errors}
