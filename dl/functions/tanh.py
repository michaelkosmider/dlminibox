import numpy as np
from dl.graph import Node


def tanh(X):
    # Compute output of module.
    exp_2x = np.exp(2 * X)
    output = (exp_2x - 1) / (exp_2x + 1)

    # Create and connect node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["output"] = output

    output.node.connect(input_nodes, tanh_backward, backward_fn_params)

    return output


# Backward definition.
def tanh_backward(params, dY):
    # dX
    return [dY * (1 - params["output"] ** 2)]
