import numpy as np
from dl.graph import Variable


def flatten(X, start_axis=1, end_axis=-1):
    # Compute output of module.

    oldshape = X.data.shape

    if end_axis < 0:
        end_axis = len(oldshape) - end_axis
    if start_axis < 0:
        start_axis = len(oldshape) - start_axis

    newshape = (
        oldshape[:start_axis]
        + (np.prod(oldshape[start_axis : end_axis + 1]),)
        + oldshape[end_axis + 1 :]
    )
    Y_data = np.reshape(X.data, newshape=newshape)
    Y = Variable(Y_data)

    # Connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if X.node.propagate_grad or X.node.keep_grad:
        input_nodes.append(X.node)
        backward_fn_params["X_shape"] = oldshape

    Y.node.connect(input_nodes, flatten_backward, backward_fn_params)

    return Y


def flatten_backward(params, dY):
    G_XY = np.reshape(dY, params["X_shape"])
    return [G_XY]
