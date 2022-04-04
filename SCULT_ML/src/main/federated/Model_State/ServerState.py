import tensorflow as tf
import tensorflow_federated as tff
import attr


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
    model_weights = attr.ib()
    optimizer_state = attr.ib()
    round_num = attr.ib()


@attr.s(eq=False, frozen=True, slots=True)
class BroadcastMessage(object):
    model_weights = attr.ib()
    round_num = attr.ib()


@tf.function
def server_update(model, server_optimizer, server_state, weights_delta):
    model_weights = model.weights
    tff.utils.assign(model_weights, server_state.model_weights)
    tff.utils.assign(server_optimizer.variables(), server_state.optimizer_state)

    # Apply the update to the model.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
        tf.nest.flatten(model_weights.trainable))
    server_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Create a new state based on the updated model.
    val=tff.utils.update_state(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)
    return tff.utils.update_state(
        server_state,
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=server_state.round_num + 1)


@tf.function
def build_server_broadcast_message(server_state):
    bd = BroadcastMessage(model_weights=server_state.model_weights, round_num=server_state.round_num)
    return bd


