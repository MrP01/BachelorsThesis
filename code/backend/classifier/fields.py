import base64
from rest_framework import serializers


class Base64BinaryField(serializers.Field):
    """Base64 Field for binary data"""

    def to_representation(self, value):
        return base64.encodebytes(value).decode()

    def to_internal_value(self, data):
        return base64.decodebytes(data.strip().encode())
