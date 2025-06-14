import numpy as np
from dl.graph import Node


# incomplete
def add(A, B):
    # Compute output of module.
    output = np.add(A, B)

    # Create node in computation graph.
    output.node = Node()

    input_nodes = []
    backward_fn_params = {}

    if A.node.propagate_grad or A.node.keep_grad:
        pass

    if B.node.propagate_grad or B.node.keep_grad:
        pass

    output.node.connect(input_nodes, add_backward, backward_fn_params)

    return output


# Backward definition.
def add_backward(params, upstream):
    grads = []

    # dA

    # dB

    return grads
