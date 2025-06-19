from .node import Node


class Variable:
    def __init__(self, data, keep_grad=False):
        self.data = data
        self.node = Node(keep_grad=keep_grad)

    def backward(self):
        self.node.backward()

    def grad(self):
        return self.node.grad

    def clear_grad(self):
        self.node.clear_grad()


# Allows users set attributes of class Parameter within a module, which are automatically added to _params dict of module.
# This way, users can have untracked Variable attributes.
class Parameter(Variable):
    pass
