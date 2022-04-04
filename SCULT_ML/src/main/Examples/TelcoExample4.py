import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import os
from absl import app
from src.main import Parameters
import numpy as np

def main(args):
    csv_url = "https://docs.google.com/spreadsheets/d/1eJo2yOTVLPjcIbwe8qSQlFNpyMhYj-xVnNVUTAhwfNU/gviz/tq?tqx=out:csv"

    df = pd.read_csv(csv_url, na_values=("?",))

    client_id_colname = 'native.country'  # the column that represents client ID
    SHUFFLE_BUFFER = 1000
    NUM_EPOCHS = 1

    # split client id into train and test clients
    client_ids = df.get(client_id_colname)
    train_client_ids = client_ids.sample(frac=0.5).tolist()
    test_client_ids = [x for x in client_ids if x not in train_client_ids]

    def create_tf_dataset_for_client_fn(client_id):
        # a function which takes a client_id and returns a
        # tf.data.Dataset for that client
        client_data = df[df[client_id_colname] == client_id]
        dataset = tf.data.Dataset.from_tensor_slices(client_data.to_dict('list'))
        dataset = dataset.shuffle(SHUFFLE_BUFFER).batch(1).repeat(NUM_EPOCHS)
        return dataset

    train_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=train_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )
    test_data = tff.simulation.ClientData.from_clients_and_fn(
        client_ids=test_client_ids,
        create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
    )

    example_dataset = train_data.create_tf_dataset_for_client(
        train_data.client_ids[0]
    )
    #print(type(example_dataset))
    #example_element = iter(example_dataset.element_spec).next()
    #print(example_element)
    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]
    def retrieve_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(2, return_sequences=True),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        return model
    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=retrieve_model(),
            input_spec=example_dataset.element_spec,
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
