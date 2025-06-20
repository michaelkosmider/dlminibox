import numpy as np
from dl.graph import Variable


def sum(X, axis=None):
    # Compute output of module.
    Y_data = np.sum(X.data, axis=axis)
    Y = Variable(Y_data)

    # Connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["axis"] = axis
        backward_fn_params["input_shape"] = X.data.shape

    Y.node.connect(input_nodes, sum_backward, backward_fn_params)

    return Y


# Backward definition.
def sum_backward(params, dY):
    # dX
    if dY is not None:
        expanded_upstream = np.expand_dims(dY, params["axis"])
        dX = np.broadcast_to(expanded_upstream, params["input_shape"])

    else:
        dX = np.ones(params["input_shape"])
    return [dX]
