import numpy as np
from dl import Module
from dl.graph import Node


class Relu(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # Compute output of module.
        output = np.maximum(0, X)

        # Create node in computation graph.
        output.node = Node()

        input_nodes = []
        backward_fn_params = {}

        if X.node.propagate_grad or X.node.keep_grad:
            input_nodes.append(X.node)
            backward_fn_params["output"] = output

        output.node.connect(input_nodes, Relu.relu_backward, backward_fn_params)

        return output

    # Backward definition.
    @staticmethod
    def relu_backward(params, upstream):
        # dX
        return [
            upstream * (1 * (params["output"] > 0))
        ]  # the 1 * might not be necessary
