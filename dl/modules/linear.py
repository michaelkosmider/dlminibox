import numpy as np
from dl import Parameter, Module
from dl.functions import matmul, add


class Linear(Module):
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
        X = matmul(X, self.W)
        X = add(X, self.b)

        return X
