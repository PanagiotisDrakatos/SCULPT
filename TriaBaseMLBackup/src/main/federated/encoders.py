from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
import tensorflow as tf


class Encode:
    def __init__(self):
        pass

    @staticmethod
    def broadcast_encoder_fn(value):
        spec = tf.TensorSpec(value.shape, value.dtype)
        if value.shape.num_elements() > 10000:
            return te.encoders.as_simple_encoder(
                te.encoders.uniform_quantization(bits=8), spec)
        else:
            return te.encoders.as_simple_encoder(te.encoders.identity(), spec)

    @staticmethod
    def mean_encoder_fn(value):
        spec = tf.TensorSpec(value.shape, value.dtype)
        if value.shape.num_elements() > 10000:
            return te.encoders.as_gather_encoder(
                te.encoders.uniform_quantization(bits=8), spec)
        else:
            return te.encoders.as_gather_encoder(te.encoders.identity(), spec)
