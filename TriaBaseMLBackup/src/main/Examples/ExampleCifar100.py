import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import app

from src.main import Parameters
from src.main.Channel.fabric_client import Notification
from src.main.Dataset.Cifar100 import Cifar
from src.main.Dataset.DatasetLoader import DatasetFactory, LoadDataset
from src.main.Utils import Util
from src.main.federated.Model_State import ClientState


def main(args):
    notification = Notification("localhost", 50051)
    factory = DatasetFactory(LoadDataset(Cifar()))
    train_data, test_data = factory.load_dataset()

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=factory.retrieve_model(True),
            input_spec=test_data.element_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, Parameters.server_adam_optimizer_fn, Parameters.client_adam_optimizer_fn)
    server_state = iterative_process.initialize()

    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    model = tff_model_fn()

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
        server_state, metrics = iterative_process.next(server_state, sampled_train_data)
        train_metrics = metrics['train']
        print(metrics)
        """ server_state_encode = Util.encode64(server_state)
                sampled_train_data_encode = Util.encode64(sampled_train_data)
                server_state_decode = Util.decode64(server_state_encode)
                sampled_train_data_decode = Util.decode64(sampled_train_data_encode)
                if server_state.__eq__(server_state_decode):
                    print("true")
                if sampled_train_data.__eq__(sampled_train_data_decode):
                    print("true")"""


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
