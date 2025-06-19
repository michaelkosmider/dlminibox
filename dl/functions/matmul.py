import numpy as np
from dl.graph import Variable


def matmul(A, B):
    # Compute output of function.
    Y_data = np.matmul(A.data, B.data)
    Y = Variable(Y_data)

    # Connect node in computation graph.

    input_nodes = []
    backward_fn_params = {}

    if A.node.propagate_grad or A.node.keep_grad:
        input_nodes.append(A.node)
        backward_fn_params["B"] = B.data

    if B.node.propagate_grad or B.node.keep_grad:
        input_nodes.append(B.node)
        backward_fn_params["A"] = A.data

    Y.node.connect(input_nodes, matmul_backward, backward_fn_params)

    return Y


def matmul_backward(params, upstream=None):
    grads = []

    # dA
    if "B" in params:
        grads.append(upstream @ params["B"].T)

    # dB
    if "A" in params:
        grads.append(params["A"].T @ upstream)

    return grads
