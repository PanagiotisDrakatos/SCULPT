import collections

import tensorflow as tf
import tensorflow_federated as tff
import functools
from src.main import Parameters


class Emnist:

    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self):
        emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=True)

        def element_fn(element):
            return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])

        def preprocess_train_dataset(dataset):
            return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
                count=Parameters.FLAGS.client_epochs_per_round).batch(
                Parameters.FLAGS.batch_size, drop_remainder=False)

        def preprocess_test_dataset(dataset):
            return dataset.map(element_fn).batch(Parameters.FLAGS.test_batch_size, drop_remainder=False)

        emnist_train = emnist_train.preprocess(preprocess_train_dataset)
        emnist_test = preprocess_test_dataset(emnist_test.create_tf_dataset_from_all_clients())

        return emnist_train, emnist_test

    def retrieve_model(self, val):
        data_format = 'channels_last'
        input_shape = [28, 28, 1]
        max_pool = functools.partial(
            tf.keras.layers.MaxPooling2D,
            pool_size=(2, 2),
            padding='same',
            data_format=data_format)
        conv2d = functools.partial(
            tf.keras.layers.Conv2D,
            kernel_size=5,
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu)

        model = tf.keras.models.Sequential([
            conv2d(filters=32, input_shape=input_shape),
            max_pool(),
            conv2d(filters=64),
            max_pool(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10 if val else 62),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        return model
