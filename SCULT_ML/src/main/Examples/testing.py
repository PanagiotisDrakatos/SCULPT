
from absl import app
from src.main.Utils import Util
from src.main.Channel.fabric_client import Notification
from src.main.Channel.federated_pb2 import FederatedRequest
import time


def main(args):
 notification = Notification("localhost", 50051)
 notification.notify(FederatedRequest(timestamp=str(time.time()),
                                      round=str("round_num"),
                                      server_state=Util.encode_bytes64("server_state"),
                                      sampled_train_data=Util.encode_bytes64("sampled_train_data"),
                                      clients_participated=str("train_data.client_ids"),
                                      broadcasted_bits=str("broadcasted_bits"),
                                      aggregated_bits=str("aggregated_bits")))
 return;


if __name__ == '__main__':
    app.run(main)


def start():
    app.run(main)
