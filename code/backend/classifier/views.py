from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import zmq

from .serializers import *


context = zmq.Context()
print("Connecting to classifier...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


class ClassifyAPIView(APIView):
    def post(self, request):
        serializer = ClassificationRequestSerializer(data=request.data)
        if serializer.is_valid():
            encrypted = serializer.data["encrypted"]
            socket.send_json()
            result = socket.recv_json()
            return Response(result)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
