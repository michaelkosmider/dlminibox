class SGD:
    def __init__(self, parameters, learning_rate):
        self.learning_rate = learning_rate
        self.parameters = parameters

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()

    def update_parameters(self):
        for param in self.parameters:
            param.data -= self.learning_rate * param.grad()
