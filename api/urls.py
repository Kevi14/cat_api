"""
URL configuration for cat_api project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.schemas import get_schema_view
from config import API_MOUNT_PATH
from .views import *
def trigger_error(request):
    division_by_zero = 1 / 0

urlpatterns = [
    path('cat-breed/', CatBreedIdentificationView.as_view()),
    path('cat-breed-gallery/', CatBreedIdentificationGalleryView.as_view()),

]
