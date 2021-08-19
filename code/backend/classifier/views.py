from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import zmq
# TODO: import messagepack?

from .serializers import *


context = zmq.Context()
print("Connecting to classifier...")
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")


class ClassifyPlainAPIView(APIView):
    def post(self, request):
        serializer = PlainClassificationRequestSerializer(data=request.data)
        if serializer.is_valid():
            while True:
                try:
                    socket.send_json({
                        "action": "predict_plain",
                        "image": serializer.data["image"]
                    })
                    result = socket.recv_json()
                    break
                except zmq.ZMQError:
                    socket.connect("tcp://localhost:5555")
            return Response(result)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class ClassifyEncryptedAPIView(APIView):
    def post(self, request):
        serializer = EncryptedClassificationRequestSerializer(data=request.data)
        if serializer.is_valid():
            while True:
                try:
                    socket.send_json({
                        "action": "predict_encrypted",
                        "relinKey": serializer.data["relinKey"],
                        "galoisKey": serializer.data["galoisKey"],
                        "ciphertext": serializer.data["ciphertext"],
                    })
                    result = socket.recv_json()
                    break
                except zmq.ZMQError:
                    socket.connect("tcp://localhost:5555")
            return Response(result)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
