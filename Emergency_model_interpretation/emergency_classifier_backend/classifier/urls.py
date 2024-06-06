# urls.py (inside the 'classifier' app)

from django.urls import path
from .views import predict_emergency

urlpatterns = [
    path('predict/', predict_emergency, name='predict'),
]
