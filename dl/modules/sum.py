import numpy as np
from dl.graph import Node


# refactor to module
def sum(X, axis=None):
    # Compute output of module.
    output = np.sum(X, axis=axis)

    # Create node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["axis"] = axis
        backward_fn_params["input_shape"] = X.shape
        backward_fn_params["requires_upstream"] = len(output.shape) > 1

    output.node.connect(input_nodes, sum_backward, backward_fn_params)

    return output


# Backward definition.
def sum_backward(params, upstream):
    # dX
    if upstream is not None:
        dX = np.ones(params["input_shape"]) * np.expand_dims(upstream, params["axis"])
    else:
        if params["requires_upstream"]:
            print(
                "error, requires upstream"
            )  # check this error later. Ensure output is a scalar in this case.
        dX = np.ones(params["input_shape"])
    return [dX]
