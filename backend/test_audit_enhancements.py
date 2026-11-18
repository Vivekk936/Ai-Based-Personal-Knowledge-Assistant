import os
import shutil
import tempfile
import json
import unittest

os.environ.setdefault("ADA_DATA_DIR", os.path.join(tempfile.gettempdir(), "ada_audit_test_data"))
shutil.rmtree(os.environ["ADA_DATA_DIR"], ignore_errors=True)

import config
import store
import chunker
from tools import read_chunk, read_chunks, search_documents, TOOL_SCHEMAS, READ_ONLY
from harness import (
    format_attached_prompt,
    _tool_call_from_text,
    _call_tool,
    _manage_context,
    _partial_answer,
    SYSTEM_PROMPT,
)


class TestAuditEnhancements(unittest.TestCase):
    def setUp(self):
        conn = store.connect()
        conn.close()

    def test_configurable_environment_variables(self):
        self.assertGreaterEqual(config.AGENT_MAX_STEPS, 20)
        self.assertGreaterEqual(config.AGENT_WALL_CLOCK_SECONDS, 300)
        self.assertGreaterEqual(config.AGENT_MAX_TOOL_CALLS, 30)
        self.assertGreaterEqual(config.MAX_UPLOAD_BYTES, 50 * 1024 * 1024)
        self.assertGreaterEqual(config.PDF_PAGE_BATCH_SIZE, 10)
        self.assertGreaterEqual(config.CHUNK_SIZE, 500)
        self.assertGreaterEqual(config.CHUNK_OVERLAP, 50)
        self.assertGreaterEqual(config.EMBED_BATCH_SIZE, 8)

    def test_read_chunks_tool(self):
        test_path = os.path.join(tempfile.gettempdir(), "test_doc.pdf")
        chunks = [
            {"id": "doc1_p1_c0", "path": test_path, "filename": "test_doc.pdf", "page": 1, "text": "Content of passage zero."},
            {"id": "doc1_p1_c1", "path": test_path, "filename": "test_doc.pdf", "page": 1, "text": "Content of passage one."},
            {"id": "doc1_p2_c0", "path": test_path, "filename": "test_doc.pdf", "page": 2, "text": "Content of passage two."},
        ]
        store.replace_chunks(test_path, chunks)


        single = read_chunk("doc1_p1_c0")
        self.assertIn("Content of passage zero.", single)
        self.assertIn("test_doc.pdf", single)


        multi = read_chunks(["doc1_p1_c0", "doc1_p2_c0"])
        self.assertIn("Content of passage zero.", multi)
        self.assertIn("Content of passage two.", multi)
        self.assertIn("test_doc.pdf:1", multi)
        self.assertIn("test_doc.pdf:2", multi)


        empty_call = read_chunks([])
        self.assertIn("No chunk IDs provided", empty_call)

        missing = read_chunks(["non_existent_id"])
        self.assertIn("found in the index", missing)

    def test_incremental_append_chunks(self):
        test_path = os.path.join(tempfile.gettempdir(), "large_doc.pdf")

        batch1 = [
            {"id": "ld_p1_c0", "path": test_path, "filename": "large_doc.pdf", "page": 1, "text": "Page 1 specification."},
        ]
        batch2 = [
            {"id": "ld_p2_c0", "path": test_path, "filename": "large_doc.pdf", "page": 2, "text": "Page 2 hydraulic limits."},
        ]


        store.append_chunks(test_path, batch1, clear_first=True)
        retrieved1 = store.get_chunks(["ld_p1_c0"])
        self.assertEqual(len(retrieved1), 1)


        store.append_chunks(test_path, batch2, clear_first=False)
        retrieved2 = store.get_chunks(["ld_p1_c0", "ld_p2_c0"])
        self.assertEqual(len(retrieved2), 2)
        self.assertEqual(retrieved2["ld_p1_c0"]["text"], "Page 1 specification.")
        self.assertEqual(retrieved2["ld_p2_c0"]["text"], "Page 2 hydraulic limits.")

    def test_lexical_search_with_filename_filter(self):
        doc_a = os.path.join(tempfile.gettempdir(), "alpha.pdf")
        doc_b = os.path.join(tempfile.gettempdir(), "beta.pdf")

        store.replace_chunks(doc_a, [
            {"id": "a_1", "path": doc_a, "filename": "alpha.pdf", "page": 1, "text": "Centrifugal pump motor 75kW."}
        ])
        store.replace_chunks(doc_b, [
            {"id": "b_1", "path": doc_b, "filename": "beta.pdf", "page": 1, "text": "Centrifugal pump impeller diameter 220mm."}
        ])


        hits_all = store.search_lexical("centrifugal", k=10)
        cids_all = [cid for cid, _ in hits_all]
        self.assertIn("a_1", cids_all)
        self.assertIn("b_1", cids_all)


        hits_a = store.search_lexical("centrifugal", k=10, filename="alpha.pdf")
        cids_a = [cid for cid, _ in hits_a]
        self.assertIn("a_1", cids_a)
        self.assertNotIn("b_1", cids_a)

    def test_attachment_prompt_injection(self):
        prompt = format_attached_prompt(
            "What is the torque rating?",
            ["motor_specs.pdf", "gearbox.docx"]
        )
        self.assertIn("<attached_documents>", prompt)
        self.assertIn("motor_specs.pdf", prompt)
        self.assertIn("gearbox.docx", prompt)
        self.assertIn("What is the torque rating?", prompt)


        unchanged = format_attached_prompt("Simple query", [])
        self.assertEqual(unchanged, "Simple query")
        unchanged_none = format_attached_prompt("Simple query", None)
        self.assertEqual(unchanged_none, "Simple query")

    def test_tool_call_recovery_and_execution(self):
        call = _call_tool("read_chunks", '{"chunk_ids": ["doc1_p1_c0"]}')
        self.assertIn("Content of passage zero.", call)


        tool_names = [t["function"]["name"] for t in TOOL_SCHEMAS]
        self.assertIn("read_chunks", tool_names)
        self.assertIn("search_documents", tool_names)

    def test_chunker_configurable_sizing(self):
        text = "Sentence one. " * 150
        chunks = chunker.chunk_page(text, target_chars=400, overlap_chars=80)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(len(c), 600)


if __name__ == "__main__":
    unittest.main()
