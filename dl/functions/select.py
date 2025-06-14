import numpy as np
from dl import Variable
from dl.graph import Node


def select(X, *indices):

    # Compute output of module.
    if len(indices) == 1:
        indices = indices[0]

    output = X[indices]

    # Can happen if output is a scalar.
    if not isinstance(output, Variable):
        output = Variable(output)

    # Create and connect node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.keep_grad or X.node.propagate_grad:
        input_nodes.append(X.node)
        backward_fn_params["shape"] = X.shape
        backward_fn_params["indices"] = indices

    output.node.connect(input_nodes, select_backward, backward_fn_params)

    return output


def select_backward(params, upstream=None):

    if upstream is None:
        upstream = 1.0

    grad = np.zeros(params["shape"], dtype=np.float64)
    np.add.at(grad, params["indices"], upstream)

    return [grad]
