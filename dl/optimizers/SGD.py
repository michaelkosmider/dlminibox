from .optimizer import Optimizer
import numpy as np


class SGD(Optimizer):
    def __init__(
        self, parameters, learning_rate=0.01, weight_decay=1e-4, momentum=None
    ):
        super().__init__()

        self.parameters = parameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.momentum = momentum

        if momentum is not None:
            self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def update_parameters(self):

        if self.momentum is not None:

            for param, velocity in zip(self.parameters, self.velocities):

                if param.grad is None:
                    continue

                # The actual gradient at the current weights.
                grad = param.grad + self.weight_decay * param.data

                # Update the velocity based on the gradient.
                velocity *= self.momentum
                velocity -= self.learning_rate * grad

                # Update parameters based on new velocity.
                param.data += velocity

        else:

            for param in self.parameters:

                if param.grad is None:
                    continue

                grad = param.grad + self.weight_decay * param.data

                param.data -= self.learning_rate * grad
