from dl import Module
from dl.functions import relu
from dl.functions import sigmoid
from dl.functions import flatten
from dl.functions import tanh


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return relu(X)


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return sigmoid(X)


class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return tanh


class Flatten(Module):
    def __init__(self, start_axis=1, end_axis=-1):
        super().__init__()
        self.start_axis = start_axis
        self.end_axis = end_axis

    def forward(self, X):
        return flatten(X, self.start_axis, self.end_axis)
