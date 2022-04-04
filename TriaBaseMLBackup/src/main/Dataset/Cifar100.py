import collections
import functools

import tensorflow as tf
import tensorflow_federated as tff

from src.main import Parameters
import nest_asyncio
nest_asyncio.apply()

class Cifar:

    def __init__(self) -> None:
        super().__init__()

    def get_dataset(self):
        cifar_train, cifar_test = tff.simulation.datasets.cifar100.load_data()

        def element_fn(element):
            return collections.OrderedDict(
                x=collections.OrderedDict(
                    a=element['image']),
                y=element['label'])

        def preprocess_train_dataset(dataset):
            return dataset \
                .map(element_fn).shuffle(buffer_size=418) \
                .shuffle(buffer_size=418) \
                .repeat(count=Parameters.FLAGS.client_epochs_per_round) \
                .batch(Parameters.FLAGS.batch_size, drop_remainder=False)

        def preprocess_test_dataset(dataset):
            return dataset.map(element_fn).batch(Parameters.FLAGS.test_batch_size, drop_remainder=False)

        cifar_train = cifar_train.preprocess(preprocess_train_dataset)
        cifar_test = preprocess_test_dataset(cifar_test.create_tf_dataset_from_all_clients())

        return cifar_train, cifar_test

    def retrieve_model(self, val):
        data_format = 'channels_last'
        input_shape = [32, 32, 3]
        max_pool = functools.partial(
            tf.keras.layers.MaxPooling2D,
            pool_size=(2, 2),
            padding='same',
            data_format=data_format)
        conv2d = functools.partial(
            tf.keras.layers.Conv2D,
            kernel_size=(5,5),
            use_bias=False,
            strides=2,
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu)

        model = tf.keras.models.Sequential([
            conv2d(filters=96, input_shape=input_shape),
            max_pool(),
            conv2d(filters=192),
            max_pool(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        return model
