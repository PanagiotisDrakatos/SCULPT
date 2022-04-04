import tensorflow as tf
import tensorflow_federated as tff
from absl import app

from src.main import Parameters
from src.main.Dataset.DatasetLoader import DatasetFactory, LoadDataset
from src.main.Dataset.Shakespeare import Shakespeares

vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r')


class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

    def __init__(self, name='accuracy', dtype=tf.float32):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [-1, 1])
        y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
        return super().update_state(y_true, y_pred, sample_weight)


def main(args):
    factory = DatasetFactory(LoadDataset(Shakespeares()))

    train_data, test_data = factory.load_dataset()
    keras_model = factory.retrieve_model(True)

    def tff_model_fn():
        input_spec = test_data.element_spec
        keras_model_clone = tf.keras.models.clone_model(keras_model)
        return tff.learning.from_keras_model(
            keras_model_clone,
            input_spec=input_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[FlattenedCategoricalAccuracy()])

    iterative_process = tff.learning.build_federated_averaging_process(tff_model_fn,
                                                                       Parameters.server_adam_optimizer_fn,
                                                                       Parameters.client_adam_optimizer_fn)
    server_state = iterative_process.initialize()
    server_state = tff.learning.state_with_new_model_weights(
        server_state,
        trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
        non_trainable_weights=[
            v.numpy() for v in keras_model.non_trainable_weights
        ])


    def data(client, source=train_data):
        return factory.load_instance().process(train_data,client).take(5)

    clients = [
        'ALL_S_WELL_THAT_ENDS_WELL_CELIA', 'MUCH_ADO_ABOUT_NOTHING_OTHELLO','THE_TRAGEDY_OF_KING_LEAR_KING',
    ]

    train_datasets = [data(client) for client in clients]
    test_dataset = tf.data.Dataset.from_tensor_slices(
        [data(client, test_data) for client in clients]).flat_map(lambda x: x)

    # We concatenate the test datasets for evaluation with Keras by creating a
    # Dataset of Datasets, and then identity flat mapping across all the examples.
    test_dataset = tf.data.Dataset.from_tensor_slices(
        [data(client, test_data) for client in clients]).flat_map(lambda x: x)

    for round_num in range(Parameters.FLAGS.total_rounds):
        print('Round {r}'.format(r=round_num))

        # keras_evaluate(server_state, Parameters.FLAGS.total_rounds)
        state, metrics = iterative_process.next(server_state, train_datasets)
        train_metrics = metrics['train']
        print('\tTrain: loss={l:.3f}, accuracy={a:.3f}'.format(
            l=train_metrics['loss'], a=train_metrics['accuracy']))


if __name__ == '__main__':
    app.run(main)
