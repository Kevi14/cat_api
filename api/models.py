from django.db import models
from django.contrib.auth import get_user_model


User = get_user_model()

class CatBreed(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(default="")

class Photo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='photos/')
    cat_breed = models.ForeignKey(CatBreed, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

