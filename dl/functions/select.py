import numpy as np
from dl import Variable


def select(X, *indices):
    # Compute output of module.
    if len(indices) == 1:
        indices = indices[0]

    Y_data = X.data[indices]
    Y = Variable(Y_data)

    # Connect node in computation graph.

    input_nodes = []
    backward_fn_params = {}

    if X.node.keep_grad or X.node.propagate_grad:
        input_nodes.append(X.node)
        backward_fn_params["shape"] = X.data.shape
        backward_fn_params["indices"] = indices

    Y.node.connect(input_nodes, select_backward, backward_fn_params)

    return Y


def select_backward(params, dY=None):
    if dY is None:
        dY = 1.0

    grad = np.zeros(params["shape"], dtype=np.float64)
    np.add.at(grad, params["indices"], dY)

    return [grad]
