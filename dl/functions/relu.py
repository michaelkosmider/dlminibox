import numpy as np
from dl.graph import Node


def relu(X):

    # Compute output of module.
    output = np.maximum(0, X)

    # Create and connect node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["output"] = output

    output.node.connect(input_nodes, relu_backward, backward_fn_params)

    return output


def relu_backward(params, upstream):
    dX = [upstream * (1 * (params["output"] > 0))]
    return dX
