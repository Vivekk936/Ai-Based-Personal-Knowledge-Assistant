import json
import logging
from django.conf import settings
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
import httpx

log = logging.getLogger("ada.django")
BACKEND_URL = getattr(settings, "ADA_BACKEND_URL", "http://127.0.0.1:8000")


# ---------------------------------------------------------------------------
# Template Views
# ---------------------------------------------------------------------------

def index(request):
    """Main Chat Interface."""
    return render(request, "chat.html", {
        "backend_url": BACKEND_URL,
        "page_title": "Chat",
    })


def organize(request):
    """File Organizer Interface."""
    return render(request, "organize.html", {
        "backend_url": BACKEND_URL,
        "page_title": "Smart File Organizer",
    })


def eval_view(request):
    """RAG Evaluation Benchmark Dashboard."""
    return render(request, "eval.html", {
        "backend_url": BACKEND_URL,
        "page_title": "RAG Evaluation Benchmark",
    })


# ---------------------------------------------------------------------------
# Backend Proxy APIs
# ---------------------------------------------------------------------------

def _proxy_get(path: str, params=None):
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{BACKEND_URL}{path}", params=params)
            return JsonResponse(resp.json(), status=resp.status_code, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Backend unavailable: {e}"}, status=503)


def _proxy_post(path: str, data=None):
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(f"{BACKEND_URL}{path}", json=data)
            return JsonResponse(resp.json(), status=resp.status_code, safe=False)
    except Exception as e:
        return JsonResponse({"error": f"Backend error: {e}"}, status=503)


def health(request):
    return _proxy_get("/health")


@csrf_exempt
def sessions(request):
    if request.method == "POST":
        return _proxy_post("/sessions")
    return _proxy_get("/sessions")


@csrf_exempt
def session_detail(request, session_id):
    if request.method == "GET":
        return _proxy_get(f"/sessions/{session_id}/messages")
    elif request.method == "PATCH":
        body = json.loads(request.body) if request.body else {}
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.patch(f"{BACKEND_URL}/sessions/{session_id}", json=body)
                return JsonResponse(resp.json(), status=resp.status_code)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=503)
    elif request.method == "DELETE":
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.delete(f"{BACKEND_URL}/sessions/{session_id}")
                return JsonResponse(resp.json(), status=resp.status_code)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=503)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def chat(request):
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    body = json.loads(request.body) if request.body else {}
    return _proxy_post("/chat", body)


@csrf_exempt
def chat_stream(request):
    """SSE streaming proxy from FastAPI /chat/stream."""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)
    body = json.loads(request.body) if request.body else {}

    def event_stream():
        try:
            with httpx.Client(timeout=300.0) as client:
                with client.stream("POST", f"{BACKEND_URL}/chat/stream", json=body) as resp:
                    for chunk in resp.iter_bytes():
                        yield chunk
        except Exception as e:
            err_event = json.dumps({"t": "error", "message": f"Stream error: {e}"})
            yield f"data: {err_event}\n\n".encode("utf-8")

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


@csrf_exempt
def folders(request):
    if request.method == "DELETE":
        folder = request.GET.get("folder", "")
        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.delete(f"{BACKEND_URL}/folders", params={"folder": folder})
                return JsonResponse(resp.json(), status=resp.status_code)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=503)
    return _proxy_get("/folders")


@csrf_exempt
def folders_pick(request):
    return _proxy_post("/folders/pick")


@csrf_exempt
def folders_refresh(request):
    body = json.loads(request.body) if request.body else {}
    return _proxy_post("/folders/refresh", body)


def index_status(request):
    folder = request.GET.get("folder")
    return _proxy_get("/index/status", params={"folder": folder} if folder else None)


@csrf_exempt
def ingest(request):
    if request.method == "GET":
        return _proxy_get("/ingest")
    if request.method == "POST":
        files = []
        for f in request.FILES.getlist("files"):
            files.append(("files", (f.name, f.read(), f.content_type or "application/octet-stream")))
        try:
            with httpx.Client(timeout=180.0) as client:
                resp = client.post(f"{BACKEND_URL}/ingest", files=files)
                return JsonResponse(resp.json(), status=resp.status_code)
        except Exception as e:
            return JsonResponse({"error": f"Ingest error: {e}"}, status=503)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def ingest_detail(request, filename):
    if request.method == "DELETE":
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.delete(f"{BACKEND_URL}/ingest/{filename}")
                return JsonResponse(resp.json(), status=resp.status_code)
        except Exception as e:
            return JsonResponse({"error": f"Delete error: {e}"}, status=503)
    return JsonResponse({"error": "Method not allowed"}, status=405)


@csrf_exempt
def apply_moves(request):
    body = json.loads(request.body) if request.body else {}
    return _proxy_post("/apply", body)


@csrf_exempt
def eval_rag(request):
    pdf = request.GET.get("pdf")
    if request.method == "POST":
        return _proxy_post(f"/eval/rag{f'?pdf={pdf}' if pdf else ''}")
    return _proxy_get(f"/eval/rag{f'?pdf={pdf}' if pdf else ''}")


@csrf_exempt
def eval_rag_upload(request):
    if request.method != "POST" or "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)
    uploaded = request.FILES["file"]
    try:
        with httpx.Client(timeout=180.0) as client:
            files = {"file": (uploaded.name, uploaded.read(), uploaded.content_type)}
            resp = client.post(f"{BACKEND_URL}/eval/rag/upload", files=files)
            return JsonResponse(resp.json(), status=resp.status_code)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=503)
