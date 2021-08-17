from rest_framework import serializers
from .fields import Base64BinaryField


class PlainClassificationRequestSerializer(serializers.Serializer):
    image = serializers.ListField(
        child=serializers.IntegerField(min_value=0, max_value=255)  # TODO: 0..1
    )


class EncryptedClassificationRequestSerializer(serializers.Serializer):
    relinKey = Base64BinaryField()
    galoisKey = Base64BinaryField()
    encryptedImage = Base64BinaryField()
