import threading
import time

import grpc
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from absl import app

from src.main import Parameters
from src.main.Channel.fabric_client import Notification
from src.main.Channel.federated_pb2 import FederatedRequest
from src.main.Dataset.DatasetLoader import DatasetFactory, LoadDataset
from src.main.Utils import Util, StringBuilder
from src.main.federated import Computation, ServiceThread
from src.main.federated.Model_State import ClientState
from src.main.federated.Wrapper import KerasModelWrapper
from tensorflow.python.client import device_lib
from src.main.Dataset.Smarty import Smarty
from src.main.federated.Compression import Compression

def make_remote_executor(inferred_cardinalities):
    """Make remote executor."""

    def create_worker_stack(ex):
        ex = tff.framework.ThreadDelegatingExecutor(ex)
        return tff.framework.ReferenceResolvingExecutor(ex)

    client_ex = []
    num_clients = inferred_cardinalities.get(tff.CLIENTS, None)
    if num_clients:
        print('Inferred that there are {} clients'.format(num_clients))
    else:
        print('No CLIENTS placement provided')

    for _ in range(num_clients or 0):
        channel = grpc.insecure_channel('{}:{}'.format(Parameters.FLAGS.host, Parameters.FLAGS.port))
        remote_ex = tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING')
        worker_stack = create_worker_stack(remote_ex)
        client_ex.append(worker_stack)

    federating_strategy_factory = tff.framework.FederatedResolvingStrategy.factory(
        {
            tff.SERVER: create_worker_stack(tff.framework.EagerTFExecutor()),
            tff.CLIENTS: client_ex,
        })
    unplaced_ex = create_worker_stack(tff.framework.EagerTFExecutor())
    federating_ex = tff.framework.FederatingExecutor(federating_strategy_factory,
                                                     unplaced_ex)
    return tff.framework.ReferenceResolvingExecutor(federating_ex)


def main(args):
    print(device_lib.list_local_devices())
    x = threading.Thread(target=ServiceThread.thread_init, name='server', daemon=True)
    x.start()
    notification = Notification("localhost", 50051)
    factory = DatasetFactory(LoadDataset(Smarty()))
    train_data, test_data = factory.load_dataset()

    def tff_model_fn():
        """Constructs a fully initialized model for use in federated averaging."""
        keras_model = factory.retrieve_model(True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return KerasModelWrapper(keras_model, test_data.element_spec, loss)

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
    str_loss = StringBuilder()
    str_acc = StringBuilder()
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
        broadcasted_bits, aggregated_bits = evaluate(round_num, train_metrics, server_state, model, environment,
                                                     test_data, metric,str_acc,str_loss)
        print(sampled_train_data)
        """ server_state_encode = Util.encode64(server_state)
        sampled_train_data_encode = Util.encode64(sampled_train_data)
        server_state_decode = Util.decode64(server_state_encode)
        sampled_train_data_decode = Util.decode64(sampled_train_data_encode)
        if server_state.__eq__(server_state_decode):
            print("true")
        if sampled_train_data.__eq__(sampled_train_data_decode):
            print("true")"""

        notification.notify(FederatedRequest(timestamp=str(time.time()),
                                             round=str(round_num),
                                             server_state=Util.encode_bytes64(server_state),
                                             sampled_train_data=Util.encode_bytes64(sampled_train_data),
                                             clients_participated=str(train_data.client_ids),
                                             broadcasted_bits=str(broadcasted_bits),
                                             aggregated_bits=str(aggregated_bits)))

    print(str_loss)
    print(str_acc)


def evaluate(round_num, train_metrics, server_state, model, environment, test_data, metric, str_acc,str_loss):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, test_data, metric)
        str_acc.append("(" + str(round_num) + "," + format(accuracy) + ")")
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))
    return broadcasted_bits, aggregated_bits

def evaluateRAW(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, metric)
        str_acc.append("(" + str(round_num) + "," + format(accuracy) + ")")
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))

    return broadcasted_bits, aggregated_bits


def evaluateCompression(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    Compression.write_data_to_files(aggregated_bits, Parameters.file_name1)

    # zip the file_name to zip_file_name
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, metric)
        str_acc.append("(" + str(round_num) + "," + format(accuracy) + ")")
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))
    Compression.file_compress(Parameters.file_name_list, Parameters.zip_file_name)


def evaluateSampling(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss, dataset):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    Compression.write_data_to_files(aggregated_bits, Parameters.file_name1)
    start = time.time()
    sampling_dataset = tf.data.Dataset.sample_from_datasets([dataset], weights=[Parameters.weights, Parameters.weights])
    sampling_time = time.time()
    # zip the file_name to zip_file_name
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, metric)
        str_acc.append("(" + str(round_num) + "," + format(accuracy) + ")")
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))
    print(list(sampling_dataset.as_numpy_iterator()))
    return sampling_time, broadcasted_bits, aggregated_bits


def evaluateRandom(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss, dataset):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    Compression.write_data_to_files(aggregated_bits, Parameters.file_name1)
    start = time.time()
    random_dataset = tf.data.Dataset.random_from_datasets.random(seed=Parameters.seed).take(dataset.size)
    random_time = time.time()
    # zip the file_name to zip_file_name
    print('round {:2d}, training loss={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num, train_metrics,
                                                                                          Util.format_size(
                                                                                              broadcasted_bits),
                                                                                          Util.format_size(
                                                                                              aggregated_bits)))
    if round_num % Parameters.FLAGS.rounds_per_eval == 0:
        model.from_weights(server_state.model_weights)
        accuracy = ClientState.keras_evaluate(model.keras_model, metric)
        str_acc.append("(" + str(round_num) + "," + format(accuracy) + ")")
        print('round {:2d}, validation accuracy={}, broadcasted_bits={}, aggregated_bits={}'.format(round_num,
                                                                                                    accuracy * 100.0,
                                                                                                    Util.format_size(
                                                                                                        broadcasted_bits),
                                                                                                    Util.format_size(
                                                                                                        aggregated_bits)))
    print(list(random_dataset.as_numpy_iterator()))
    return random_time, broadcasted_bits, aggregated_bits



if __name__ == '__main__':
    app.run(main)


def start():
    app.run(main)
