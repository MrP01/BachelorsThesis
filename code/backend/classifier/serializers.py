from rest_framework import serializers
from .fields import Base64BinaryField


class ClassificationRequestSerializer(serializers.Serializer):
    encrypted = Base64BinaryField()
    parameter1 = serializers.IntegerField()
