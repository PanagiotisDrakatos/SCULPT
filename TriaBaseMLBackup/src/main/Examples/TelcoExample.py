import collections
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from absl import app
from tensorflow.keras import layers
from src.main.Utils import Util, StringBuilder
from src.main import Parameters
from src.main.federated.Model_State import ClientState
from src.main.federated.Compression import Compression

global input_spec
import random
import time


def main(args):
    working_dir = "D:/User/Documents/GitHub/TriaBaseMLBackup/input/fakehdfs/nms/ystr=2016/ymstr=1/ymdstr=28"
    client_id_colname = 'counter'
    SHUFFLE_BUFFER = 1000
    NUM_EPOCHS = 1

    for root, dirs, files in os.walk(working_dir):
        file_list = []

        for filename in files:
            if filename.endswith('.csv'):
                file_list.append(os.path.join(root, filename))
        df_list = []
        for file in file_list:
            df = pd.read_csv(file, delimiter="|", usecols=[1, 2, 6, 7], header=None, na_values=["NIL"],
                             na_filter=True, names=["time", "meas_info", "counter", "value"], encoding='latin-1')
            # df_list.append(df[["value"]])

        if df_list:
            rawdata = pd.concat(df_list)

    # print(df.head())
    client_ids = df.get(client_id_colname)
    train_client_ids = client_ids.sample(frac=0.5).tolist()

    # test_client_ids = [x for x in client_ids if x not in train_client_ids]
    # test_client_ids = [x for x in client_ids if x not in train_client_ids]

    def create_tf_dataset_for_client_fn(client_id):
        # a function which takes a client_id and returns a
        # tf.data.Dataset for that client
        # target = df.pop('value')
        client_data = df[df['value'] == client_id]
        # print(df.head())
        features = ['time', 'meas_info', 'value']
        LABEL_COLUMN = 'counter'
        dataset = tf.data.Dataset.from_tensor_slices(
            (collections.OrderedDict(client_data[features].to_dict('list')),
             client_data[LABEL_COLUMN].to_list())
        )
        global input_spec
        input_spec = dataset.element_spec
        dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
        return dataset

    train_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    # test_data = tff.simulation.ClientData.from_clients_and_fn(
    #    client_ids=test_client_ids,
    #    create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    # )
    example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0]
    )

    # split client id into train and test clients
    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    def retrieve_model():
        SEQUENCE_LENGTH = 5
        features = ['time', 'meas_info', 'value']
        input_dict = {f: tf.keras.layers.Input(shape=(SEQUENCE_LENGTH, 1), name=f) for f in features}
        concatenated_inputs = tf.keras.layers.Concatenate()(input_dict.values())
        lstm_output = tf.keras.layers.LSTM(2, input_shape=(1, 2), return_sequences=True)(concatenated_inputs)
        logits = tf.keras.layers.Dense(256, activation=tf.nn.relu)(lstm_output)
        predictions = tf.keras.layers.Activation(tf.nn.softmax)(logits)
        model = tf.keras.models.Model(inputs=input_dict, outputs=predictions)
        return model

    print(input_spec)

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=retrieve_model(),
            input_spec=example_dataset.element_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, Parameters.server_adam_optimizer_fn, Parameters.client_adam_optimizer_fn)
    server_state = iterative_process.initialize()
    environment = Util.info()
    str_loss = StringBuilder()
    str_acc = StringBuilder()
    # metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    metric = tf.keras.metrics.RootMeanSquaredError(name='room mean square error');
    model = tff_model_fn()
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
        # train_metrics = metrics['train']
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        # broadcasted_bits, aggregated_bits = evaluate(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss)


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
    start = time.time()
    Compression.write_data_to_files(aggregated_bits, Parameters.file_name1)
    Compression.file_compress(Parameters.file_name_list, Parameters.zip_file_name)
    compression_time = time.time() - start
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
    return broadcasted_bits, aggregated_bits,compression_time
#Creates a dataset by interleaving elements of datasets with weight[i] probability of picking an element from dataset i.
# Sampling is done without replacement. For example, suppose we have 2 datasets:
#A dataset that interleaves elements from datasets at random, according to weights if provided, otherwise with uniform probability
def evaluateSampling(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss, dataset):
    size_info = environment.get_size_info()
    broadcasted_bits = size_info.broadcast_bits[-1]
    aggregated_bits = size_info.aggregate_bits[-1]
    str_loss.append("(" + str(round_num) + "," + str(train_metrics) + ")")
    Compression.write_data_to_files(aggregated_bits, Parameters.file_name1)
    start = time.time()
    sampling_dataset = tf.data.Dataset.sample_from_datasets([dataset], weights=[Parameters.weights, Parameters.weights])
    sampling_time = time.time()-start
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
    start = 0  # inclusive
    end = dataset.size  # exclusive
    n = 1
    x = np.random.uniform(low=start, high=end, size=(n)).astype(int)
    random_dataset = tf.data.Dataset.random_from_datasets.random(seed=Parameters.seed).range(x)
    random_time = time.time()-start
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
