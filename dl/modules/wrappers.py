from dl import Module
from dl.functions import relu
from dl.functions import sigmoid


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
