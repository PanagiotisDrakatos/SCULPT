import collections
import tensorflow_federated as tff

ModelWeights = collections.namedtuple('ModelWeights', 'trainable non_trainable')
ModelOutputs = collections.namedtuple('ModelOutputs', 'loss')


class KerasModelWrapper(object):

    def __init__(self, keras_model, input_spec, loss):
        self.keras_model = keras_model
        self.input_spec = input_spec
        self.loss = loss

    def forward_pass(self, batch_input, training=True):
        preds = self.keras_model(batch_input['x'], training=training)
        loss = self.loss(batch_input['y'], preds)
        return ModelOutputs(loss=loss)

    @property
    def weights(self):
        return ModelWeights(
            trainable=self.keras_model.trainable_variables,
            non_trainable=self.keras_model.non_trainable_variables)

    def from_weights(self, model_weights):
        tff.utils.assign(self.keras_model.trainable_variables,
                         list(model_weights.trainable))
        tff.utils.assign(self.keras_model.non_trainable_variables,
                         list(model_weights.non_trainable))

    @property
    def get_keras_model(self):
        return self.keras_model
