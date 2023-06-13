from rest_framework import serializers
from .models import CatBreed, Photo
from django.conf import settings
class CatBreedSerializer(serializers.ModelSerializer):
    class Meta:
        model = CatBreed
        fields = '__all__'

class PhotoSerializer(serializers.ModelSerializer):
    image_path = serializers.SerializerMethodField()

    class Meta:
        model = Photo
        fields = ['id', 'image', 'cat_breed', 'image_path']

    def get_image_path(self, obj):
        request = self.context.get('request')
        if obj.image and request is not None:
            base_url = request.build_absolute_uri('/')[:-1]
            print(base_url)
            return "localhost:8000" + obj.image.url
        return None