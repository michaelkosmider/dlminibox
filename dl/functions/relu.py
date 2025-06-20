import numpy as np
from dl.graph import Variable


def relu(X):
    # Compute output of module.
    Y_data = np.maximum(0, X.data)
    Y = Variable(Y_data)

    # Connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["Y"] = Y.data

    Y.node.connect(input_nodes, relu_backward, backward_fn_params)

    return Y


def relu_backward(params, dY):
    G_XY = dY * (1 * (params["Y"] > 0))
    return [G_XY]
