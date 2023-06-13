from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import CatBreed, Photo
from .serializers import CatBreedSerializer, PhotoSerializer
from ai.use_classifier import find_cat_breed


class CatBreedIdentificationView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        image_file = request.FILES.get('image')
        if not image_file:
            return Response({'error': 'Image file is required.'}, status=status.HTTP_400_BAD_REQUEST)
# TODO : enable real cat breed
        # breed_id = find_cat_breed(image_file)

        try:
            cat_breed = CatBreed.objects.get(id=7)
            serializer = CatBreedSerializer(cat_breed)
            return Response({"breed":serializer.data}, status=status.HTTP_200_OK)
        except CatBreed.DoesNotExist:
            return Response({'error': 'Cat breed not found.'}, status=status.HTTP_404_NOT_FOUND)


class CatBreedIdentificationGalleryView(APIView):
    permission_classes = [IsAuthenticated]

    def get_queryset(self, request):
        queryset = Photo.objects.filter(user=request.user)

        # Additional filters
        breed_id = request.GET.get('breed_id')
        if breed_id:
            queryset = queryset.filter(cat_breed__id=breed_id)

        # Add more filters as needed

        return queryset

    def get(self, request):
        queryset = self.get_queryset(request)

        serializer = PhotoSerializer(queryset, many=True, context={'request': request})

        return Response(serializer.data, status=status.HTTP_200_OK)

    def post(self, request):
        image_file = request.FILES.get('image')
        breed_id = request.data.get('breed_id')  # Assuming the breed ID is provided in the request data

        if not image_file:
            return Response({'error': 'Image file is required.'}, status=status.HTTP_400_BAD_REQUEST)

        breed_name = find_cat_breed(image_file)

        try:
            cat_breed = CatBreed.objects.get(id=breed_id)
        except CatBreed.DoesNotExist:
            return Response({'error': 'Cat breed not found.'}, status=status.HTTP_404_NOT_FOUND)

        photo = Photo(user=request.user, image=image_file, cat_breed=cat_breed)
        photo.save()

        serializer = PhotoSerializer(photo)
        return Response(serializer.data, status=status.HTTP_200_OK)
