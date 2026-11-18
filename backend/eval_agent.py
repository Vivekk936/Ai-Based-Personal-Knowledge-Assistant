import os
import re
import sys
import time

from config import OUTPUT_DIR
from harness import _CANNOT

CASES = [
    {
        "id": "grounded-pressure",
        "q": "What is the maximum operating pressure mentioned for pump P-101? Cite the source.",
        "check": lambda r: bool(r["citations"]) and "PSI" in r["answer"],
        "want": "cites a source and reports a pressure",
    },
    {
        "id": "exact-code",
        "q": "Which document mentions drawing DWG-11015-A? Give the filename.",
        "check": lambda r: ".txt" in r["answer"] or ".pdf" in r["answer"] or ".md" in r["answer"],
        "want": "names a filename",
    },
    {
        "id": "abstain-absent",


        "q": "What is the warranty period in months for the Model Z9000 turbine?",
        "check": lambda r: not re.search(
            r"\b\d+\s*(?:-|\s)?\s*(?:month|year|day|week)s?\b", r["answer"], re.I
        ),
        "want": "states no warranty duration (none exists in the corpus)",
    },
    {
        "id": "no-fabricated-citation",
        "q": "Summarise what the documents say about bearing vibration limits, with citations.",
        "check": lambda r: r["unverified"] == 0,
        "want": "every citation traces to a real search result",
    },
    {


        "id": "deliverable-docx-prose",
        "q": "Write an application in a Word file asking for a late fee waiver — my "
             "salary arrives late and I can't afford the extra charges.",
        "check": lambda r: _docx_has_prose(35),
        "want": "a real .docx on disk containing an actual letter",
    },
    {


        "id": "no-false-refusal",
        "q": "Create a Word document with a short note about quarterly planning.",
        "check": lambda r: not _CANNOT.search(r["answer"]) and bool(r["artifacts"]),
        "want": "writes the file instead of refusing or asking what to put in it",
    },
    {
        "id": "deliverable-xlsx",
        "q": "Make an Excel file called evalparts.xlsx with columns Part and Qty, "
             "containing BOLT-M12 40 and BOLT-M8 25.",
        "check": lambda r: _xlsx_has_rows("evalparts.xlsx", [("BOLT-M12", 40), ("BOLT-M8", 25)]),
        "want": "a real .xlsx on disk with the right rows",
    },
]


def _docx_has_prose(min_words: int) -> bool:
    from docx import Document

    for name in os.listdir(OUTPUT_DIR):
        if not name.lower().endswith(".docx"):
            continue
        doc = Document(os.path.join(OUTPUT_DIR, name))
        words = sum(len(par.text.split()) for par in doc.paragraphs)
        if words >= min_words:
            return True
    return False


def _xlsx_has_rows(name: str, expected: list[tuple]) -> bool:
    import openpyxl

    path = os.path.join(OUTPUT_DIR, name)
    if not os.path.exists(path):
        return False
    wb = openpyxl.load_workbook(path)
    try:
        cells = {(str(r[0]), r[1]) for r in wb.active.iter_rows(min_row=2, values_only=True) if r[0]}
    finally:
        wb.close()
    return all((p, q) in cells for p, q in expected)


def run_case(case: dict) -> dict:
    from harness import run_agent_stream

    seen, answer, artifacts, reason, steps = set(), "", [], "", 0
    unverified_events = 0
    t0 = time.perf_counter()
    for ev in run_agent_stream(case["q"], session_id="eval"):
        if ev["t"] == "verify" and ev.get("unverified"):
            unverified_events += len(ev["unverified"])
        elif ev["t"] == "artifact":
            artifacts.append(ev["summary"])
        elif ev["t"] == "done":
            answer, reason, steps = ev["answer"], ev["reason"], ev["steps"]
            seen = set(ev["citations"])
    return {
        "answer": answer,
        "citations": sorted(seen),
        "artifacts": artifacts,
        "unverified": unverified_events,
        "reason": reason,
        "steps": steps,
        "seconds": time.perf_counter() - t0,
    }


def main(repeats: int = 3) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows, total_pass, total = [], 0, 0
    for case in CASES:
        passes, times = 0, []
        for _ in range(repeats):
            try:

                if case["id"].startswith("deliverable") or case["id"] == "no-false-refusal":
                    for stale in os.listdir(OUTPUT_DIR):
                        if stale.lower().endswith((".docx", ".xlsx")):
                            os.remove(os.path.join(OUTPUT_DIR, stale))
                r = run_case(case)
                passes += bool(case["check"](r))
                times.append(r["seconds"])
            except Exception as e:
                print(f"  {case['id']}: ERROR {e}")
                times.append(0.0)
        total_pass += passes
        total += repeats
        rows.append((case["id"], passes, repeats, sum(times) / max(len(times), 1), case["want"]))
        print(f"  {case['id']:26s} {passes}/{repeats}  {sum(times)/max(len(times),1):5.1f}s avg   {case['want']}")

    print(f"\nTOTAL {total_pass}/{total} ({total_pass/total*100:.0f}%) over {repeats} repeats")
    return rows


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
