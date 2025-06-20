import numpy as np
from dl.graph import Variable


def sigmoid(X):
    # Compute output of module.
    Y_data = 1 / (1 + np.exp(-X))
    Y = Variable(Y_data)

    # Connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["output"] = Y.data

    Y.node.connect(input_nodes, sigmoid_backward, backward_fn_params)

    return Y


# Backward definition.
def sigmoid_backward(params, dY):
    # dX
    return [dY * (params["output"] * (1 - params["output"]))]


"""

if y = sigmoid(x) then dy/dx = y(1-y)
so dL/dx = upstream * dy/dx

"""
