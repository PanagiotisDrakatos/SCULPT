from absl import flags
import tensorflow as tf

flags.DEFINE_integer('total_rounds', 40, 'Number of total training rounds.')
flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
flags.DEFINE_integer('train_clients_per_round', 10, 'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 4, 'Number of epochs in the client to take per round.')
flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')
flags.DEFINE_integer('test_batch_size', 100, 'Minibatch size of test data.')

# Client hyperparameters
flags.DEFINE_string('host', 'localhost', 'The host to connect to.')
flags.DEFINE_string('port', '8000', 'The port to connect to.')
flags.DEFINE_integer('ServerPort', 8000, 'The port to connect to.')
flags.DEFINE_integer('n_clients', 5, 'Number of clients.')
flags.DEFINE_integer('n_rounds', 3, 'Number of rounds.')

# Optimizer configuration (this defines one or more flags per optimizer).
flags.DEFINE_float('server_learning_rate', 1.0, 'Server learning rate.')
flags.DEFINE_float('client_learning_rate', 0.1, 'Client learning rate.')

# Server Variables
flags.DEFINE_integer('threads', '1', 'number of worker threads in thread pool')
flags.DEFINE_string('private_key', '', 'the private key for SSL/TLS setup')
flags.DEFINE_string('certificate_chain', '', 'the cert for SSL/TLS setup')
flags.DEFINE_integer('clients', '3', 'number of clients to host on this worker')
flags.DEFINE_integer('fanout', '100', 'max fanout in the hierarchy of local executors')

FLAGS = flags.FLAGS


def server_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.server_learning_rate)


def client_optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=FLAGS.client_learning_rate)


def client_adam_optimizer_fn():
    return tf.keras.optimizers.Adam(learning_rate=FLAGS.client_learning_rate)


def server_adam_optimizer_fn():
    return tf.keras.optimizers.Adam(learning_rate=FLAGS.server_learning_rate)


file_name1 = "temporary_file1_for_zip.csv"
file_name2 = "temporary_file2_for_zip.csv"
file_name_list = [file_name1, file_name2]
zip_file_name = "temporary.zip"
weights=0.5
seed=4

