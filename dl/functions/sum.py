import numpy as np
from dl.graph import Node


def sum(X, axis=None):
    # Compute output of module.
    output = np.sum(X, axis=axis)

    # Create and connect node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["axis"] = axis
        backward_fn_params["input_shape"] = X.shape

    output.node.connect(input_nodes, sum_backward, backward_fn_params)

    return output


# Backward definition.
def sum_backward(params, upstream):
    # dX
    if upstream is not None:
        expanded_upstream = np.expand_dims(upstream, params["axis"])
        dX = np.broadcast_to(expanded_upstream, params["input_shape"])

    else:
        dX = np.ones(params["input_shape"])
    return [dX]
