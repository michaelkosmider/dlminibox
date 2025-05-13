import numpy as np
from .softmax import softmax
from dl.graph import Node


# refactor to module
def cross_entropy_loss(X, y):
    # Compute output of module.
    probs = softmax(
        X
    )  # computed in two steps, because probs is an intermediate value needed for the backward pass.
    output = np.average(-np.log(probs[np.arange(y.shape[0]), y]))

    # Create node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if X.node.keep_grad or X.node.propagate_grad:
        input_nodes.append(X.node)
        backward_fn_params["y"] = y
        backward_fn_params["probs"] = probs

    output.node.connect(input_nodes, cross_entropy_loss_backward, backward_fn_params)

    return output


# Backward definition.
def cross_entropy_loss_backward(params, upstream=None):
    # dX
    params["probs"][np.arange(params["y"].shape[0]), params["y"]] -= 1
    params["probs"] /= params["y"].shape[0]

    if upstream is None:
        return [params["probs"]]
    else:
        return [params["probs"] * upstream]
