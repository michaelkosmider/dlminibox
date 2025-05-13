import numpy as np
from dl.graph import Node


# refactor to module
def sigmoid(X):
    # Compute output of module.
    output = 1 / (1 + np.exp(-X))

    # Create node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["output"] = output

    output.node.connect(input_nodes, sigmoid_backward, backward_fn_params)

    return output


# Backward definition.
def sigmoid_backward(params, upstream):
    # dX
    return [upstream * (params["output"] * (1 - params["output"]))]


"""

if y = sigmoid(x) then dy/dx = y(1-y)
so dL/dx = upstream * dy/dx

"""
