"""URL routing for Ada Django Frontend."""
from django.urls import path
from . import views

urlpatterns = [
    # Main Views
    path('', views.index_view, name='index'),
    path('chat/', views.chat_view, name='chat_index'),
    path('chat/<str:session_id>/', views.chat_view, name='chat_session'),
    path('organizer/', views.organizer_view, name='organizer'),
    path('evaluation/', views.evaluation_view, name='evaluation'),

    # Deliverables Download
    path('deliverables/<str:filename>/', views.download_deliverable, name='download_deliverable'),

    # JSON & SSE Proxy Endpoints
    path('api/health/', views.api_health, name='api_health'),
    path('api/index/status/', views.api_index_status, name='api_index_status'),
    path('api/sessions/', views.api_sessions, name='api_sessions'),
    path('api/sessions/<str:session_id>/', views.api_session_detail, name='api_session_detail'),
    path('api/sessions/<str:session_id>/messages/', views.api_session_messages, name='api_session_messages'),
    path('api/chat/stream/', views.api_chat_stream, name='api_chat_stream'),
    path('api/ingest/', views.api_ingest, name='api_ingest'),
    path('api/ingest/<str:filename>/', views.api_ingest_delete, name='api_ingest_delete'),
    path('api/folders/', views.api_folders, name='api_folders'),
    path('api/folders/pick/', views.api_folders_pick, name='api_folders_pick'),
    path('api/folders/refresh/', views.api_folders_refresh, name='api_folders_refresh'),
    path('api/apply/', views.api_apply, name='api_apply'),
    path('api/eval/', views.api_eval_rag, name='api_eval_rag'),
    path('api/eval/upload/', views.api_eval_upload, name='api_eval_upload'),
    path('api/deliverables/', views.api_deliverables, name='api_deliverables'),
]

