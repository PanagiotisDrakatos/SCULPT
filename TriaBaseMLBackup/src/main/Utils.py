from typing import Iterable
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te
import base64
import sys
import jsonpickle
import io


class StringBuilder(object):

    def __init__(self):
        self._stringio = io.StringIO()

    def __str__(self):
        return self._stringio.getvalue()

    def append(self, *objects, sep=' ', end=''):
        print(*objects, sep=sep, end=end, file=self._stringio)
class Util:
    def __init__(self):
        self.__callbackMap = {}
        for k in (getattr(self, x) for x in dir(self)):
            if hasattr(k, "bind_to_event"):
                self.__callbackMap.setdefault(k.bind_to_event, []).append(k)
            elif hasattr(k, "bind_to_event_list"):
                for j in k.bind_to_event_list:
                    self.__callbackMap.setdefault(j, []).append(k)

    def __eq__(self, o: object) -> bool:
        return super().__eq__(o)

    def __dir__(self) -> Iterable[str]:
        return super().__dir__()

    @staticmethod
    def encode64(data_val):
        strformat = jsonpickle.encode(data_val)
        byte = bytes(strformat, 'utf-8')
        return base64.b64encode(byte)

    @staticmethod
    def decode64(encoded_value):
        # byte = bytes(encoded_value, 'utf-8')
        decoded = base64.b64decode(encoded_value)
        strformat = decoded.decode("utf-8")
        return jsonpickle.decode(strformat)

    @staticmethod
    def encode_bytes64(data_val):
        data = str(data_val)
        byte = bytes(data, 'utf-8')
        return base64.b64encode(byte)

    @staticmethod
    def decode_bytes64(encoded_value):
        # byte = bytes(encoded_value, 'utf-8')
        decoded = base64.b64decode(encoded_value)
        return decoded.decode("utf-8")

    @staticmethod
    def broadcast_encoder_fn(value):
        """Function for building encoded broadcast."""
        spec = tf.TensorSpec(value.shape, value.dtype)
        if value.shape.num_elements() > 10000:
            return te.encoders.as_simple_encoder(
                te.encoders.uniform_quantization(bits=8), spec)
        else:
            return te.encoders.as_simple_encoder(te.encoders.identity(), spec)

    @staticmethod
    def mean_encoder_fn(value):
        """Function for building encoded mean."""
        spec = tf.TensorSpec(value.shape, value.dtype)
        if value.shape.num_elements() > 10000:
            return te.encoders.as_gather_encoder(
                te.encoders.uniform_quantization(bits=8), spec)
        else:
            return te.encoders.as_gather_encoder(te.encoders.identity(), spec)

    @staticmethod
    def format_size(size):
        """A helper function for creating a human-readable size."""
        size = float(size)
        for unit in ['bit', 'Kibit', 'Mibit', 'Gibit']:
            if size < 1024.0:
                return "{size:3.2f}{unit}".format(size=size, unit=unit)
            size /= 1024.0
        return "{size:.2f}{unit}".format(size=size, unit='TiB')

    @staticmethod
    def info():
        sizing_factory = tff.framework.sizing_executor_factory()
        context = tff.framework.ExecutionContext(executor_fn=sizing_factory)
        tff.framework.set_default_context(context)
        return sizing_factory

    @staticmethod
    def class_for_name(module_name, class_name):
        # load the module, will raise ImportError if module cannot be loaded
        m = importlib.import_module(module_name)
        # get the class, will raise AttributeError if class cannot be found
        c = getattr(m, class_name)
        return c
