import collections
import time

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import app

from src.main import Parameters
from src.main.Channel.federated_pb2 import FederatedRequest
from src.main.Dataset.DatasetLoader import DatasetFactory, LoadDataset
from src.main.Dataset.Fashion_mnist import FashionMnist
from src.main.Utils import Util
from src.main.federated import Computation
from src.main.federated.Model_State import ClientState

def main(args):

    factory = DatasetFactory(LoadDataset(FashionMnist()))
    train_data, test_data = factory.load_dataset()
    input_spec = collections.OrderedDict(x=(tf.TensorSpec(shape=(28,28), dtype=tf.float32)), y=tf.TensorSpec(shape=(None), dtype=tf.int32))

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=factory.retrieve_model(True),
            input_spec=input_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    iterative_process = Computation.build_federated_averaging_process(
        tff_model_fn, Parameters.server_optimizer_fn, Parameters.client_optimizer_fn)
    server_state = iterative_process.initialize()

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = tff_model_fn()

    # exec=Executor()
    # factory = tff.framework.ResourceManagingExecutorFactory(make_remote_executor)
    # context = tff.framework.ExecutionContext(factory)
    # tff.framework.set_default_context(context)
    environment = Util.info()
    for round_num in range(Parameters.FLAGS.total_rounds):
        sampled_clients = np.random.choice(
            train_data.client_ids,
            size=Parameters.FLAGS.train_clients_per_round,
            replace=False)
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients
        ]
        server_state, train_metrics = iterative_process.next(server_state, sampled_train_data)



def evaluate(round_num, train_metrics, server_state, model, environment, test_data, metric, notification):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, test_data, metric)
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))
    return broadcasted_bits, aggregated_bits


if __name__ == '__main__':
    app.run(main)

def start():
    app.run(main)
