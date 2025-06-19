import numpy as np
from dl.functions.softmax import softmax
from dl.graph import Variable


def cross_entropy_loss(X, y):
    # Compute output of module.
    probs = softmax(X.data)
    Y_data = np.average(-np.log(probs[np.arange(y.data.shape[0]), y.data]))
    Y = Variable(Y_data)

    # Connect node in computation graph.
    input_nodes = []
    backward_fn_params = {}

    if X.node.keep_grad or X.node.propagate_grad:
        input_nodes.append(X.node)
        backward_fn_params["y"] = y.data
        backward_fn_params["probs"] = probs

    Y.node.connect(input_nodes, cross_entropy_loss_backward, backward_fn_params)

    return Y


# Backward definition.
def cross_entropy_loss_backward(params, upstream=None):
    # dX
    params["probs"][np.arange(params["y"].shape[0]), params["y"]] -= 1
    params["probs"] /= params["y"].shape[0]  # Divide by N

    if upstream is None:
        return [params["probs"]]
    else:
        return [params["probs"] * upstream]  # upstream is dz/dL which is a scalar
