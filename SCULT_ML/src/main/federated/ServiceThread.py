import tensorflow_federated as tff

from src.main import Parameters


def thread_init():
    executor_factory = tff.framework.local_executor_factory(num_clients=Parameters.FLAGS.clients,
                                                            max_fanout=Parameters.FLAGS.fanout, clients_per_thread=1)
    tff.framework.ExecutorService(executor_factory)
    tff.simulation.run_server(executor_factory, Parameters.FLAGS.threads, Parameters.FLAGS.ServerPort)
    print("connected")
