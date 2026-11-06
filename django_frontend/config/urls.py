"""URL Configuration for Ada Django Frontend."""
from django.urls import path, include

urlpatterns = [
    path('', include('ada_app.urls')),
]
