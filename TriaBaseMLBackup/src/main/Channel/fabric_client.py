import grpc



#python -m grpc.tools.protoc -I=. --python_out=./src/main/Channel --grpc_python_out=./src/main/Channel federated.proto
from src.main.Channel.federated_pb2_grpc import FederatedServiceStub


class Notification:

    def __init__(self, host, port):
        self.host = host
        self.port = str(port)

    def notify(self, request):
        print("handler1 with param: %s" % str(request))
        channel = grpc.insecure_channel(self.host+':'+self.port)
        stub = FederatedServiceStub(channel)
        print(stub.Search(request))
        channel.close()
        return None
