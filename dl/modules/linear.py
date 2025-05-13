import numpy as np
from dl.graph import Node
from dl import Parameter
from dl import Module


class Linear(Module):
    # Module initialization.
    def __init__(self, input_size, output_size, param_init="xavier"):
        super().__init__()

        # Store hyper_parameters for print_module()
        self._hyper_parameters["input_size"] = input_size
        self._hyper_parameters["output_size"] = output_size
        self._hyper_parameters["param_init"] = param_init

        # Bias initialized to 0.
        self.b = Parameter(np.zeros(output_size), keep_grad=True)

        # Weights are initialized using xavier initialization (more types of initialization to be added later).
        if param_init == "xavier":
            xavier_range = np.sqrt(6 / (1 + input_size + output_size))
            self.W = Parameter(
                np.random.uniform(
                    -xavier_range, xavier_range, size=(input_size, output_size)
                ),
                keep_grad=True,
            )

    def forward(self, X):
        # Compute output of module.
        output = X @ self.W + self.b

        # Create node in computation graph.
        output.node = Node()

        input_nodes = []
        backward_fn_params = {}

        if X.node.propagate_grad or X.node.keep_grad:
            input_nodes.append(X.node)
            backward_fn_params["W"] = self.W

        if self.W.node.keep_grad:
            input_nodes.append(self.W.node)
            backward_fn_params["X"] = X

        if self.b.node.keep_grad:
            input_nodes.append(self.b.node)
            backward_fn_params["db"] = True

        output.node.connect(input_nodes, Linear.backward, backward_fn_params)

        return output

    # Backward definition.
    @staticmethod
    def backward(params, upstream=None):
        grads = []

        # dX
        if "W" in params:
            grads.append(upstream @ params["W"].T)

        # dW
        if "X" in params:
            grads.append(params["X"].T @ upstream)

        # db
        if "db" in params:
            grads.append(np.sum(upstream, axis=0))

        return grads
