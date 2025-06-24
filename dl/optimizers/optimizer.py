class Optimizer:

    def clear_grad(self):
        for param in self.parameters:
            param.clear_grad()
