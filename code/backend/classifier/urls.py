from django.urls import path
from .views import *

urlpatterns = [
    path("classify/plain/", ClassifyPlainAPIView.as_view())
]
