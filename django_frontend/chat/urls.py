from django.urls import path
from . import views

urlpatterns = [
    # UI Pages
    path('', views.index, name='index'),
    path('organize/', views.organize, name='organize'),
    path('eval/', views.eval_view, name='eval'),

    # API Proxy Endpoints
    path('api/health', views.health, name='api_health'),
    path('api/sessions', views.sessions, name='api_sessions'),
    path('api/sessions/<str:session_id>', views.session_detail, name='api_session_detail'),
    path('api/chat', views.chat, name='api_chat'),
    path('api/chat/stream', views.chat_stream, name='api_chat_stream'),
    path('api/folders', views.folders, name='api_folders'),
    path('api/folders/pick', views.folders_pick, name='api_folders_pick'),
    path('api/folders/refresh', views.folders_refresh, name='api_folders_refresh'),
    path('api/index/status', views.index_status, name='api_index_status'),
    path('api/ingest', views.ingest, name='api_ingest'),
    path('api/ingest/<str:filename>', views.ingest_detail, name='api_ingest_detail'),
    path('api/apply', views.apply_moves, name='api_apply'),
    path('api/eval/rag', views.eval_rag, name='api_eval_rag'),
    path('api/eval/rag/upload', views.eval_rag_upload, name='api_eval_rag_upload'),
]

