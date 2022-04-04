import tensorflow_federated as tff
import grpc
from src.main import Parameters


class Executor:
    def __init__(self):
        pass

    def make_remote_executor(self,inferred_cardinalities):
        """Make remote executor."""

        def create_worker_stack(ex):
            ex = tff.framework.ThreadDelegatingExecutor(ex)
            return tff.framework.ReferenceResolvingExecutor(ex)

        client_ex = []
        num_clients = inferred_cardinalities.get(tff.CLIENTS, None)
        if num_clients:
            print('Inferred that there are {} clients'.format(num_clients))
        else:
            print('No CLIENTS placement provided')

            for _ in range(num_clients or 0):
                channel = grpc.insecure_channel('{}:{}'.format(Parameters.FLAGS.host, Parameters.FLAGS.port))
                remote_ex = tff.framework.RemoteExecutor(channel, rpc_mode='STREAMING')
                worker_stack = create_worker_stack(remote_ex)
                client_ex.append(worker_stack)

        federating_strategy_factory = tff.framework.FederatedResolvingStrategy.factory(
            {
                tff.SERVER: create_worker_stack(tff.framework.EagerTFExecutor()),
                tff.CLIENTS: client_ex,
            })
        unplaced_ex = create_worker_stack(tff.framework.EagerTFExecutor())
        federating_ex = tff.framework.FederatingExecutor(federating_strategy_factory,
                                                         unplaced_ex)
        return tff.framework.ReferenceResolvingExecutor(federating_ex)
