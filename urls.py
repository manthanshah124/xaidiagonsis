# predictor/urls.py
from django.urls import path
from .views import predict, health, csrf_token_view, index

urlpatterns = [
    path('', index, name='index'),
    path('predict/', predict, name='predict'),
    path('health/', health, name='health'),
    path('csrf-token/', csrf_token_view, name='csrf_token'),
]