from django.conf import settings

def ada_globals(request):
    """Provides global configuration variables across all Ada Django templates."""
    return {
        'FASTAPI_BASE_URL': getattr(settings, 'FASTAPI_BASE_URL', 'http://127.0.0.1:8000'),
    }
