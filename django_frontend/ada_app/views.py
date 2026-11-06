"""Views for Ada Django Frontend.

Renders Django templates for the main views (Chat, Document Organizer, RAG Evaluation)
and provides asynchronous proxy endpoints for real-time SSE streaming, document uploads,
folder organization, and deliverables downloading.
"""
import json
import logging
import os
from pathlib import Path
from django.conf import settings
from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

from .services.api_client import get_api_client

log = logging.getLogger('ada.frontend.views')


# =============================================================================
# HTML Template Views
# =============================================================================

def index_view(request: HttpRequest) -> HttpResponse:
    """Root redirect to active or first chat session."""
    client = get_api_client()
    sessions = client.list_sessions()
    if sessions:
        return redirect('chat_session', session_id=sessions[0].get('id', 'default'))
    return redirect('chat_index')


def chat_view(request: HttpRequest, session_id: str | None = None) -> HttpResponse:
    """Renders the AI Chat Assistant interface."""
    client = get_api_client()
    sessions = client.list_sessions()

    # If no session specified, pick first or create default
    if not session_id:
        if sessions:
            session_id = sessions[0].get('id', 'default')
        else:
            new_s = client.create_session()
            session_id = new_s.get('id', 'default')
            sessions = client.list_sessions()

    messages = client.get_messages(session_id) if session_id else []
    uploads = client.list_uploads()
    deliverables = client.list_deliverables()
    health = client.get_health()

    context = {
        'active_tab': 'chat',
        'sessions': sessions,
        'active_session_id': session_id,
        'messages': messages,
        'uploads': uploads,
        'deliverables': deliverables,
        'health': health,
    }
    return render(request, 'pages/chat.html', context)


def organizer_view(request: HttpRequest) -> HttpResponse:
    """Renders the Universal Smart File Organizer interface."""
    client = get_api_client()
    sessions = client.list_sessions()
    folders_data = client.list_folders()
    uploads = client.list_uploads()
    health = client.get_health()

    context = {
        'active_tab': 'organizer',
        'sessions': sessions,
        'active_session_id': None,
        'granted_folders': folders_data.get('granted', []),
        'index_status': folders_data.get('index', {}),
        'uploads': uploads,
        'health': health,
    }
    return render(request, 'pages/organizer.html', context)


def evaluation_view(request: HttpRequest) -> HttpResponse:
    """Renders the Dynamic RAG Evaluation Benchmark interface (zero dummy data)."""
    client = get_api_client()
    sessions = client.list_sessions()
    eval_report = client.get_eval_rag(force_refresh=False)
    health = client.get_health()

    context = {
        'active_tab': 'evaluation',
        'sessions': sessions,
        'active_session_id': None,
        'eval_report': eval_report,
        'health': health,
    }
    return render(request, 'pages/evaluation.html', context)


# =============================================================================
# JSON API & SSE Streaming Endpoints
# =============================================================================

def api_health(request: HttpRequest) -> JsonResponse:
    """Health check endpoint polled by status bar and loading screen."""
    client = get_api_client()
    return JsonResponse(client.get_health())


def api_index_status(request: HttpRequest) -> JsonResponse:
    """Returns background indexing progress and queue counts."""
    client = get_api_client()
    return JsonResponse(client.get_index_status())


@csrf_exempt
def api_sessions(request: HttpRequest) -> JsonResponse:
    """Lists all chat sessions or creates a new one."""
    client = get_api_client()
    if request.method == 'POST':
        return JsonResponse(client.create_session())
    return JsonResponse({'sessions': client.list_sessions()})


@csrf_exempt
def api_session_detail(request: HttpRequest, session_id: str) -> JsonResponse:
    """Deletes or renames a chat session."""
    client = get_api_client()
    if request.method == 'DELETE':
        return JsonResponse(client.delete_session(session_id))
    elif request.method == 'PATCH':
        try:
            body = json.loads(request.body.decode('utf-8'))
            title = body.get('title', 'New Chat')
            return JsonResponse(client.rename_session(session_id, title))
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'Method not allowed'}, status=405)


def api_session_messages(request: HttpRequest, session_id: str) -> JsonResponse:
    """Returns transcript messages for a given session."""
    client = get_api_client()
    return JsonResponse({'messages': client.get_messages(session_id)})


@csrf_exempt
def api_chat_stream(request: HttpRequest) -> StreamingHttpResponse | JsonResponse:
    """Proxies Server-Sent Events (SSE) from FastAPI /chat/stream to the browser.

    Streams tokens, tool executions, citations, and status events incrementally without buffering.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    try:
        if request.content_type == 'application/json':
            data = json.loads(request.body.decode('utf-8'))
        else:
            data = request.POST

        session_id = data.get('session_id', 'default')
        message = data.get('message', '').strip()
        attachments = data.get('attachments', [])
        if not isinstance(attachments, list):
            attachments = []

        if not message and attachments:
            message = f"Please analyze the attached document(s): {', '.join(attachments)}"

        if not message:
            return JsonResponse({'error': 'message is required'}, status=400)

        client = get_api_client()
        stream = client.stream_chat_raw(session_id, message, attachments=attachments)

        response = StreamingHttpResponse(stream, content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response

    except Exception as e:
        log.exception("api_chat_stream proxy error: %s", e)
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def api_ingest(request: HttpRequest) -> JsonResponse:
    """Uploads document files (PDF, XLSX, DOCX, etc.) or lists current indexed uploads."""
    client = get_api_client()
    if request.method == 'POST':
        uploaded_files = request.FILES.getlist('files')
        if not uploaded_files:
            # Fallback for single file input
            single = request.FILES.get('file')
            if single:
                uploaded_files = [single]

        files_data = []
        for f in uploaded_files:
            files_data.append((f.name, f.read(), f.content_type or 'application/octet-stream'))

        result = client.upload_files(files_data)
        return JsonResponse(result)

    return JsonResponse({'uploads': client.list_uploads()})


@csrf_exempt
def api_ingest_delete(request: HttpRequest, filename: str) -> JsonResponse:
    """Deletes an uploaded file and forgets its index chunks."""
    if request.method != 'DELETE':
        return JsonResponse({'error': 'DELETE required'}, status=405)
    client = get_api_client()
    return JsonResponse(client.delete_upload(filename))


@csrf_exempt
def api_folders(request: HttpRequest) -> JsonResponse:
    """Returns granted folders and index status."""
    client = get_api_client()
    return JsonResponse(client.list_folders())


@csrf_exempt
def api_folders_pick(request: HttpRequest) -> JsonResponse:
    """Invokes system folder picker dialog."""
    client = get_api_client()
    return JsonResponse(client.pick_folder())


@csrf_exempt
def api_folders_refresh(request: HttpRequest) -> JsonResponse:
    """Scans and analyzes a directory to produce organization suggestions."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        body = json.loads(request.body.decode('utf-8')) if request.body else {}
        folder = body.get('folder', '').strip()
        client = get_api_client()
        return JsonResponse(client.refresh_folder(folder))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


@csrf_exempt
def api_apply_moves(request: HttpRequest) -> JsonResponse:
    """Executes approved move operations."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    try:
        body = json.loads(request.body.decode('utf-8')) if request.body else {}
        rows = body.get('rows', [])
        client = get_api_client()
        return JsonResponse(client.apply_moves(rows))
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


# Support both naming conventions
api_apply = api_apply_moves


@csrf_exempt
def api_eval_rag(request: HttpRequest) -> JsonResponse:
    """Retrieves or triggers RAG benchmark results."""
    client = get_api_client()
    if request.method == 'POST':
        return JsonResponse(client.get_eval_rag(force_refresh=True))
    force_refresh = request.GET.get('force_refresh', '').lower() in ('true', '1')
    pdf = request.GET.get('pdf')
    return JsonResponse(client.get_eval_rag(force_refresh=force_refresh, pdf=pdf))


@csrf_exempt
def api_eval_upload(request: HttpRequest) -> JsonResponse:
    """Uploads a PDF, indexes it, and executes dynamic RAG evaluation on its content."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)
    pdf_file = request.FILES.get('file')
    if not pdf_file:
        return JsonResponse({'status': 'error', 'message': 'No file uploaded'}, status=400)
    client = get_api_client()
    content = pdf_file.read()
    return JsonResponse(client.upload_eval_pdf(pdf_file.name, content))


def api_deliverables(request: HttpRequest) -> JsonResponse:
    """Lists all produced deliverables (Excel, Word, PowerPoint, Markdown)."""
    client = get_api_client()
    return JsonResponse({'deliverables': client.list_deliverables()})


def download_deliverable(request: HttpRequest, filename: str) -> FileResponse:
    """Streams a generated deliverable file for download."""
    client = get_api_client()
    path = client.get_deliverable_path(filename)
    if not path or not path.exists():
        raise Http404(f"Deliverable '{filename}' not found.")
    return FileResponse(open(path, 'rb'), as_attachment=True, filename=filename)
