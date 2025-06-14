import numpy as np
from dl.graph import Node


def matmul(A, B):
    # Compute output of function.
    output = np.matmul(A, B)

    # Create and connect node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if A.node.propagate_grad or A.node.keep_grad:
        input_nodes.append(A.node)
        backward_fn_params["B"] = B

    if B.node.propagate_grad or B.node.keep_grad:
        input_nodes.append(B.node)
        backward_fn_params["A"] = A

    output.node.connect(input_nodes, matmul_backward, backward_fn_params)

    return output


def matmul_backward(params, upstream=None):
    grads = []

    # dA
    if "B" in params:
        grads.append(upstream @ params["B"].T)

    # dB
    if "A" in params:
        grads.append(params["A"].T @ upstream)

    return grads
