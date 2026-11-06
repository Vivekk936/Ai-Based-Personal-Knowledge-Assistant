"""FastAPI Backend Client for Ada Django Frontend.

Handles all HTTP communication, file uploads, SSE chat streaming proxy,
deliverables access, and RAG benchmark requests with error handling.
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Generator, Optional
import httpx
from django.conf import settings

log = logging.getLogger('ada.frontend.client')


class AdaApiClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or getattr(settings, 'FASTAPI_BASE_URL', 'http://127.0.0.1:8000')).rstrip('/')

    # -------------------------------------------------------------------------
    # Health & System Status
    # -------------------------------------------------------------------------

    def get_health(self) -> dict:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/health")
                return res.json()
        except Exception as e:
            return {
                "status": "offline",
                "detail": str(e),
                "model_backend_reachable": False,
                "chat_model": "Unavailable",
                "vision_model": "Unavailable",
                "embed_model": "Unavailable",
                "index": {"running": False, "pending": 0, "indexed": 0, "chunks": 0},
            }

    def get_index_status(self) -> dict:
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/index/status")
                return res.json()
        except Exception:
            return {"running": False, "pending": 0, "indexed": 0, "chunks": 0}

    # -------------------------------------------------------------------------
    # Chat Sessions & Messages
    # -------------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/sessions")
                return res.json().get("sessions", [])
        except Exception as e:
            log.warning("list_sessions failed: %s", e)
            return []

    def create_session(self) -> dict:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.post(f"{self.base_url}/sessions")
                return res.json()
        except Exception as e:
            log.error("create_session failed: %s", e)
            return {"id": "default", "title": "New Chat", "error": str(e)}

    def get_messages(self, session_id: str) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/sessions/{session_id}/messages")
                return res.json().get("messages", [])
        except Exception as e:
            log.warning("get_messages for %s failed: %s", session_id, e)
            return []

    def delete_session(self, session_id: str) -> dict:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.delete(f"{self.base_url}/sessions/{session_id}")
                return res.json()
        except Exception as e:
            return {"error": str(e)}

    def rename_session(self, session_id: str, title: str) -> dict:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.patch(f"{self.base_url}/sessions/{session_id}", json={"title": title})
                return res.json()
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # Chat Execution (Blocking & SSE Streaming)
    # -------------------------------------------------------------------------

    def send_chat(self, session_id: str, message: str, attachments: list[str] | None = None) -> dict:
        try:
            with httpx.Client(timeout=180.0) as client:
                res = client.post(
                    f"{self.base_url}/chat",
                    json={"session_id": session_id, "message": message, "attachments": attachments or []},
                )
                return res.json()
        except Exception as e:
            return {"error": str(e), "answer": f"Request failed: {e}", "unverified_citations": []}

    def stream_chat_raw(
        self, session_id: str, message: str, attachments: list[str] | None = None
    ) -> Generator[bytes, None, None]:
        """Streams Server-Sent Events (SSE) from FastAPI /chat/stream as raw byte chunks.

        Yields SSE lines (e.g. b"data: {...}\\n\\n") without buffering.
        """
        payload = {"session_id": session_id, "message": message, "attachments": attachments or []}
        try:
            with httpx.Client(timeout=600.0) as client:
                with client.stream("POST", f"{self.base_url}/chat/stream", json=payload) as resp:
                    for line in resp.iter_lines():
                        if line:
                            yield f"{line}\n\n".encode("utf-8")
                        else:
                            yield b"\n"
        except Exception as e:
            err_ev = json.dumps({"t": "error", "message": f"Connection error: {e}"})
            yield f"data: {err_ev}\n\n".encode("utf-8")

    # -------------------------------------------------------------------------
    # Document Upload & Ingestion
    # -------------------------------------------------------------------------

    def list_uploads(self) -> list[dict]:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/ingest")
                return res.json().get("uploads", [])
        except Exception as e:
            log.warning("list_uploads failed: %s", e)
            return []

    def upload_files(self, files_list: list[tuple[str, bytes, str]]) -> dict:
        """Uploads files to FastAPI: list of (filename, bytes_data, content_type)."""
        if not files_list:
            return {"ingested": []}
        multipart = [("files", (fn, b, ct or "application/octet-stream")) for fn, b, ct in files_list]
        try:
            with httpx.Client(timeout=180.0) as client:
                res = client.post(f"{self.base_url}/ingest", files=multipart)
                return res.json()
        except Exception as e:
            return {"error": str(e), "ingested": []}

    def delete_upload(self, filename: str) -> dict:
        try:
            with httpx.Client(timeout=15.0) as client:
                res = client.delete(f"{self.base_url}/ingest/{filename}")
                return res.json()
        except Exception as e:
            return {"error": str(e)}

    # -------------------------------------------------------------------------
    # File Organizer
    # -------------------------------------------------------------------------

    def list_folders(self) -> dict:
        try:
            with httpx.Client(timeout=5.0) as client:
                res = client.get(f"{self.base_url}/folders")
                return res.json()
        except Exception:
            return {"granted": [], "index": {"running": False}}

    def pick_folder(self) -> dict:
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.base_url}/folders/pick")
                return res.json()
        except Exception as e:
            return {"error": str(e), "picked": None}

    def refresh_folder(self, folder_path: str) -> dict:
        try:
            with httpx.Client(timeout=60.0) as client:
                # Ensure folder granted
                try:
                    client.post(f"{self.base_url}/folders/pick", json={"folder": folder_path})
                except Exception:
                    pass
                res = client.post(f"{self.base_url}/folders/refresh", json={"folder": folder_path})
                return res.json()
        except Exception as e:
            return {"error": str(e), "suggestions": []}

    def apply_moves(self, rows: list[dict]) -> dict:
        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.base_url}/apply", json={"rows": rows})
                return res.json()
        except Exception as e:
            return {"error": str(e), "applied": 0, "skipped": len(rows), "errors": [str(e)]}

    # -------------------------------------------------------------------------
    # RAG Evaluation Benchmark
    # -------------------------------------------------------------------------

    def get_eval_rag(self, force_refresh: bool = False, pdf: Optional[str] = None, offline: bool = False) -> dict:
        try:
            params = {}
            if force_refresh:
                params["force_refresh"] = "true"
            if pdf:
                params["pdf"] = pdf
            if offline:
                params["offline"] = "true"
            with httpx.Client(timeout=180.0) as client:
                method = client.post if force_refresh else client.get
                res = method(f"{self.base_url}/eval/rag", params=params)
                return res.json()
        except Exception as e:
            return {"status": "error", "message": f"Failed to retrieve RAG evaluation: {e}"}

    def upload_eval_pdf(self, filename: str, content: bytes) -> dict:
        try:
            with httpx.Client(timeout=180.0) as client:
                files = {"file": (filename, content, "application/pdf")}
                res = client.post(f"{self.base_url}/eval/rag/upload", files=files)
                return res.json()
        except Exception as e:
            return {"status": "error", "message": f"Upload and eval failed: {e}"}

    # -------------------------------------------------------------------------
    # Generated Deliverables
    # -------------------------------------------------------------------------

    def list_deliverables(self) -> list[dict]:
        """Scans backend/ada_data/output for produced deliverable files."""
        out_dir = Path(getattr(settings, 'DELIVERABLES_DIR', ''))
        if not out_dir.exists():
            return []
        items = []
        for p in sorted(out_dir.glob("*.*"), key=lambda f: f.stat().st_mtime, reverse=True):
            if p.is_file():
                ext = p.suffix.lower().lstrip('.')
                if ext in ('xlsx', 'docx', 'pptx', 'md', 'pdf', 'csv', 'txt'):
                    items.append({
                        "filename": p.name,
                        "size": p.stat().st_size,
                        "mtime": p.stat().st_mtime,
                        "ext": ext,
                    })
        return items

    def get_deliverable_path(self, filename: str) -> Optional[Path]:
        safe_name = os.path.basename(filename)
        out_dir = Path(getattr(settings, 'DELIVERABLES_DIR', ''))
        target = out_dir / safe_name
        if target.exists() and target.is_file():
            return target
        return None


# Global singleton helper
_client_instance = None


def get_api_client() -> AdaApiClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = AdaApiClient()
    return _client_instance
