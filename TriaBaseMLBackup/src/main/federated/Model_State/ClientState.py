import tensorflow as tf
import tensorflow_federated as tff
import attr


@attr.s(eq=False, frozen=True, slots=True)
class ClientOutput(object):
    weights_delta = attr.ib()
    client_weight = attr.ib()
    model_output = attr.ib()


@tf.function
def client_update(model, dataset, server_message, client_optimizer):
    model_weights = model.weights
    initial_weights = server_message.model_weights
    tff.utils.assign(model_weights, initial_weights)

    num_examples = tf.constant(0, dtype=tf.int32)
    loss_sum = tf.constant(0, dtype=tf.float32)

    for batch in iter(dataset):
        with tf.GradientTape() as tape:
            outputs = model.forward_pass(batch)
        grads = tape.gradient(outputs.loss, model_weights.trainable)
        grads_and_vars = zip(grads, model_weights.trainable)
        client_optimizer.apply_gradients(grads_and_vars)
        batch_size = tf.shape(batch['x'])[0]
        num_examples += batch_size
        loss_sum += outputs.loss * tf.cast(batch_size, tf.float32)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          model_weights.trainable,
                                          initial_weights.trainable)
    client_weight = tf.cast(num_examples, tf.float32)
    c = ClientOutput(weights_delta, client_weight, loss_sum / client_weight)
    return c


def keras_evaluate(model, test_data, metric):
    metric.reset_states()
    for batch in test_data:
        preds = model(batch['x'], training=False)
        metric.update_state(y_true=batch['y'], y_pred=preds)
    return metric.result()
