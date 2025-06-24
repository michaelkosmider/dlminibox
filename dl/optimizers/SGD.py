from .optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, parameters, learning_rate, weight_decay=1e-4):
        super().__init__()

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.parameters = parameters

    def update_parameters(self):
        for param in self.parameters:
            grad = param.grad() + self.weight_decay * param.data
            param.data -= self.learning_rate * grad
