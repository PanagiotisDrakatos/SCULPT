from concurrent import futures
import grpc

from src.main.Channel.federated_pb2 import FederatedResponse
from src.main.Channel.federated_pb2_grpc import FederatedServiceServicer, add_FederatedServiceServicer_to_server


class SampleServicer(FederatedServiceServicer):
    def Search(self, request, context):
        print("client message", request)
        return FederatedResponse(response="ok")


def server_init():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_FederatedServiceServicer_to_server(SampleServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print('server started')
    server.wait_for_termination(45343434)
