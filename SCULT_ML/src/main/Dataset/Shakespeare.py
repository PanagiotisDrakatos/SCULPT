import os

import tensorflow as tf
import tensorflow_federated as tff

from src.main import Parameters

vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')
SEQ_LENGTH = 100
BUFFER_SIZE = 100
BATCH_SIZE = 1


class Shakespeares:

    def __init__(self) -> None:
        super().__init__()

    def process(self, source, val):
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=vocab, values=tf.constant(list(range(len(vocab))),
                                               dtype=tf.int64)),
            default_value=0)

        def split_input_target(chunk):
            input_text = tf.map_fn(lambda x: x[:-1], chunk)
            target_text = tf.map_fn(lambda x: x[1:], chunk)
            return (input_text, target_text)

        # Construct a lookup table to map string chars to indexes,
        # using the vocab loaded above:
        table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=vocab, values=tf.constant(list(range(len(vocab))),
                                               dtype=tf.int64)),
            default_value=0)

        def to_ids(x):
            s = tf.reshape(x['snippets'], shape=[1])
            chars = tf.strings.bytes_split(s).values
            ids = table.lookup(chars)
            return ids

        def split_input_target(chunk):
            input_text = tf.map_fn(lambda x: x[:-1], chunk)
            target_text = tf.map_fn(lambda x: x[1:], chunk)
            return (input_text, target_text)

        def preprocess(dataset):
            return (
                dataset.map(to_ids)
                    .unbatch()
                    .batch(SEQ_LENGTH + 1, drop_remainder=True)
                    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
                    .repeat(count=Parameters.FLAGS.client_epochs_per_round)
                    .map(split_input_target))

        raw_example_dataset = source.create_tf_dataset_for_client(val)
        result = preprocess(raw_example_dataset)

        return result

    def get_dataset(self):

        train, test = tff.simulation.datasets.shakespeare.load_data()
        # train = train.preprocess(preprocess)
        test = self.process(train, 'THE_TRAGEDY_OF_KING_LEAR_KING')
        return train, test

    def retrieve_model(self, val):
        urls = {
            1: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch1.kerasmodel',
            8: 'https://storage.googleapis.com/tff-models-public/dickens_rnn.batch8.kerasmodel'}
        assert 1 in urls, 'batch_size must be in ' + str(urls.keys())
        url = urls[1]
        local_file = tf.keras.utils.get_file(os.path.basename(url), origin=url)
        return tf.keras.models.load_model(local_file, compile=False)
