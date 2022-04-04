import collections
import functools
import os
import tensorflow as tf
from tensorflow_federated.python.simulation import hdf5_client_data
from src.main import Parameters


class FashionMnist:

    def __init__(self) -> None:
        super().__init__()

    def __load(self, only_digits=True, cache_dir=None):
        files = ['test', 'train']

        paths = []
        for filterer in files:
            filename = filterer + '.gz'
            path = tf.keras.utils.get_file(
                filename,
                origin='https://www.kaggle.com/benedictwilkinsai/mnist-hd5f/download/',
                hash_algorithm='sha256',
                extract=True,
                archive_format='tar',
                cache_dir=cache_dir)
            paths.append(path)

        dir_path = os.path.dirname(paths[0])
        train_client_data = hdf5_client_data.HDF5ClientData(
            os.path.join(dir_path, str(paths[0]).rsplit('.', 1)[0] + '.h5'))
        test_client_data = hdf5_client_data.HDF5ClientData(
            os.path.join(dir_path, paths[1] + '.h5'))

        return train_client_data, test_client_data

    def get_dataset(self):
        emnist_train, emnist_test = tf.keras.datasets.fashion_mnist.load_data()
        def element_fn(element):
            return collections.OrderedDict(x=tf.expand_dims(element['pixels'], -1), y=element['label'])

        def preprocess_train_dataset(dataset):
            return dataset.map(element_fn).shuffle(buffer_size=418).repeat(
                count=Parameters.FLAGS.client_epochs_per_round).batch(
                Parameters.FLAGS.batch_size, drop_remainder=False)

        def preprocess_test_dataset(dataset):
            return dataset.map(element_fn).batch(Parameters.FLAGS.test_batch_size, drop_remainder=False)

       # emnist_train = emnist_train.preprocess(preprocess_train_dataset)
       # emnist_test = preprocess_test_dataset(emnist_test.create_tf_dataset_from_all_clients())
        return emnist_train, emnist_test

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
            kernel_size=(3,3),
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu)

        model = tf.keras.models.Sequential([
            conv2d(filters=64, input_shape=input_shape),
            max_pool(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation=tf.nn.relu),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Activation(tf.nn.softmax),
        ])

        return model
