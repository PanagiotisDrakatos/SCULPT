import os
import collections
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from absl import app
from tensorflow.keras import layers
import numpy as np
from src.main import Parameters
from sklearn.model_selection import train_test_split


def main(args):
    working_dir = "D:/User/Documents/GitHub/TriaBaseMLBackup/input/fakehdfs/nms/ystr=2016/ymstr=1/ymdstr=26"
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
                             na_filter=True, names=["meas_info", "counter", "value", "time"], index_col='time')
            df_list.append(df[["value"]])

        if df_list:
            rawdata = pd.concat(df_list)

    client_ids = df.get(client_id_colname)
    train_client_ids = client_ids.sample(frac=0.5).tolist()

    # test_client_ids = [x for x in client_ids if x not in train_client_ids]
    X, y = np.arange(10).reshape((5, 2)), range(5)
    dict=df.to_dict('series')
    print(df.head)
    values = df.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    test = values[n_train_hours:, :]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    train_X = tf.data.Dataset.from_tensor_slices(train_X.reshape((train_X.shape[0], 1, train_X.shape[1])))
    train_Xs=tff.simulation.FromTensorSlicesClientData(train_X)
    test_X = tf.data.Dataset.from_tensor_slices(test_X.reshape((test_X.shape[0], 1, test_X.shape[1])))
    #x_train, x_test = train_test_split(X, test_size=0.33, random_state=42)
    #x_trains = tff.simulation.FromTensorSlicesClientData(collections.OrderedDict(x_train)).batch(Parameters.FLAGS.test_batch_size, drop_remainder=False)
    #x_tests = tf.data.Dataset.from_tensor_slices(x_test)
    dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
    def element_fn(element):
        return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])

    def preprocess_train_dataset(dataset):
        return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
            count=Parameters.FLAGS.client_epochs_per_round).batch(
            Parameters.FLAGS.batch_size, drop_remainder=False)

    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(Parameters.FLAGS.test_batch_size, drop_remainder=False)

    #emnist_train = train_X.preprocess(preprocess_train_dataset)
    #emnist_test = preprocess_test_dataset(test_X.create_tf_dataset_from_all_clients())

    # split client id into train and test clients
    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

    def retrieve_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        return model


    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=retrieve_model(),
            input_spec=test_X.element_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, Parameters.server_adam_optimizer_fn, Parameters.client_adam_optimizer_fn)
    server_state = iterative_process.initialize()

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


if __name__ == '__main__':
    app.run(main)


def start():
    app.run(main)
