import numpy as np
from dl.graph import Variable


def add(A, B):
    # Compute output of module.
    Y_data = np.add(A.data, B.data)
    Y = Variable(Y_data)

    # Create and connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if A.node.propagate_grad or A.node.keep_grad:
        backward_fn_params["A_shape"] = A.data.shape
        input_nodes.append(A.node)

    if B.node.propagate_grad or B.node.keep_grad:
        backward_fn_params["B_shape"] = B.data.shape
        input_nodes.append(B.node)

    if len(input_nodes) > 0:
        backward_fn_params["Y_shape"] = Y.data.shape

    Y.node.connect(input_nodes, add_backward, backward_fn_params)

    return Y


# Backward definition.
def add_backward(params, upstream):
    grads = []

    # G_AY
    if "A_shape" in params:
        axis = deduce_axis(params["A_shape"], params["Y_shape"])
        G_AY = np.sum(upstream, axis=axis, keepdims=True)
        G_AY = np.reshape(G_AY, params["A_shape"])  # Remove padding dimensions.
        grads.append(G_AY)

    if "B_shape" in params:
        axis = deduce_axis(params["B_shape"], params["Y_shape"])
        G_BY = np.sum(upstream, axis=axis, keepdims=True)
        G_BY = np.reshape(G_BY, params["B_shape"])
        grads.append(G_BY)

    return grads


def deduce_axis(in_shape, out_shape):
    # Pad input shape with singleton dimensions.
    padding_needed = len(out_shape) - len(in_shape)
    in_shape = (1,) * padding_needed + in_shape

    axis = []
    for i in range(len(out_shape)):
        if in_shape[i] == 1 and out_shape[i] != 1:
            axis.append(i)

    return tuple(axis)
