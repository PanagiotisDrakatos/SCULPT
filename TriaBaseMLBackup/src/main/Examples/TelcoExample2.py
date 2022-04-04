import os

import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from absl import app
from tensorflow.keras import layers

from src.main import Parameters


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

    loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
    metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]


    def retrieve_model():
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      metrics=['accuracy'])
        return model


    #labels = tf.constant(['A', 'B', 'A'])  # ==> 3x1 tensor
    #tf.shape(labels)
   # target = df.pop('value')
    #dataset = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), target.values))
    batched_features = tf.constant([[[1, 3], [2, 3]],
                                    [[2, 1], [1, 2]],
                                    [[3, 3], [3, 2]]], shape=(3, 2, 2))
    batched_labels = tf.constant([['A', 'A'],
                                  ['B', 'B'],
                                  ['A', 'B']], shape=(3, 2, 1))
    t3_3D = tf.constant(value=[[[2, 3, 4]], [[4, 5, 6]]])
    print(t3_3D.shape)
    dataset = tf.data.Dataset.from_tensors(t3_3D)
    print(list(dataset))
    #dataset = tf.data.Dataset.from_tensor_slices((batched_features, batched_labels))
    input_spec = dataset.element_spec
    print(input_spec)

    def tff_model_fn() -> tff.learning.Model:
        return tff.learning.from_keras_model(
            keras_model=retrieve_model(),
            input_spec=input_spec,
            loss=loss_builder(),
            metrics=metrics_builder())

    iterative_process = tff.learning.build_federated_averaging_process(
        tff_model_fn, Parameters.server_adam_optimizer_fn, Parameters.client_adam_optimizer_fn)
    server_state = iterative_process.initialize()


if __name__ == '__main__':
    app.run(main)


def start():
    app.run(main)
