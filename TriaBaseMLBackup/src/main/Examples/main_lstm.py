import os
from absl import app

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler

from src.main import Parameters


def main(args):
    working_dir = "D:/User/Documents/GitHub/TriaBaseMLBackup/input/fakehdfs/nms/ystr=2016/ymstr=1/ymdstr=26"
    client_id_colname = 'counter'
    SHUFFLE_BUFFER = 1000
    NUM_EPOCHS = 1
    DATA_SIZE = 64
    BATCH_SIZE = 32
    for root, dirs, files in os.walk(working_dir):
        file_list = []

        for filename in files:
            if filename.endswith('.csv'):
                file_list.append(os.path.join(root, filename))
        df_list = []
        for file in file_list:
            df = pd.read_csv(file, delimiter="|", usecols=[1, 2, 6, 7],
                             header=None, na_values=["NIL"],
                             na_filter=True,
                             names=["time", "meas_info", "counter", "value"],
                             encoding='latin-1')
            df_list.append(df)

        if df_list:
            df = pd.concat(df_list)
    df = df.dropna()
    for col in df.columns:
        df[col] = df[col].astype(np.float32)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized_data = scaler.fit_transform(df.values)
    df = pd.DataFrame(normalized_data, columns=["time", "meas_info", "counter",
                                                "value"])
    client_ids = df.get(client_id_colname)
    train_client_ids = client_ids.sample(frac=0.5).tolist()

    # test_client_ids = [x for x in client_ids if x not in train_client_ids]
    # test_client_ids = [x for x in client_ids if x not in train_client_ids]

    def create_tf_dataset_for_client_fn(client_id):
        # a function which takes a client_id and returns a
        # tf.data.Dataset for that client
        # target = df.pop('value')
        client_data = df[df[client_id_colname] == client_id]
        # print(df.head())
        sample_data = client_data[['time', 'meas_info', 'value']].to_numpy()
        y = client_data[['counter']].to_numpy()
        z_dims = np.ceil(sample_data.shape[0] / DATA_SIZE).astype(int)
        samples_data = np.array_split(sample_data, z_dims)

        data_part = np.zeros((z_dims, DATA_SIZE, 3), dtype=np.float32)
        y_part = np.zeros((z_dims, DATA_SIZE, 1), dtype=np.float32)

        y_data = np.array_split(y, z_dims)

        for i in range(len(samples_data)):
            for j in range(samples_data[i].shape[0]):
                data_part[i, j, :] = samples_data[i][j, :]
        for i in range(len(y_data)):
            for j in range(y_data[i].shape[0]):
                y_part[i, j, :] = y_data[i][j, :]
        dataset = tf.data.Dataset.from_tensor_slices(
            (data_part, y_part)
        )
        dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).repeat(NUM_EPOCHS)
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

    def retrieve_model():
        input_layer = layers.Input(shape=(DATA_SIZE, 3), dtype=tf.float32)
        lstm_output = tf.keras.layers.LSTM(50, return_sequences=True)(input_layer)
        logits = tf.keras.layers.Dense(256, activation=tf.nn.relu)(lstm_output)
        predictions = tf.keras.layers.Dense(1, activation='tanh')(logits)
        model = tf.keras.models.Model(input_layer, predictions)
        print(f"model spec: {model.input_spec}")
        return model

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=retrieve_model(),
            input_spec=example_dataset.element_spec,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.Accuracy()])

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, Parameters.server_adam_optimizer_fn,
        Parameters.client_adam_optimizer_fn)
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
        server_state, metrics = iterative_process.next(server_state,
                                                       sampled_train_data)
        # train_metrics = metrics['train']
        print('round {:2d}, metrics={}'.format(round_num, metrics))
        # broadcasted_bits, aggregated_bits = evaluate(round_num, train_metrics, server_state, model, environment, metric, str_acc, str_loss)


if __name__ == '__main__':
    app.run(main)