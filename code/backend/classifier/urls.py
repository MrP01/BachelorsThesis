from django.urls import path
from .views import *

urlpatterns = [
    path("classify/", ClassifyAPIView.as_view())
]
