import numpy as np
from .node import Node


# Extending numpy arrays to have a node attribute. This code is derived from the examples in NumPy's subclassing docs.
# If only the array is passed in, then a Node object is created with keep_grad == False and no children or backward_fn.
class Variable(np.ndarray):
    def __new__(cls, input_array, keep_grad=False):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # add the new attribute to the created instance
        obj.node = Node(keep_grad=keep_grad)

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.info = getattr(obj, "info", None)

    # Alias, equivalent to self.node.backward().
    def backward(self):
        self.node.backward()

    # Alias, equivalent to self.node.grad.
    def grad(self):
        return self.node.grad

    # Alias, equivalent to self.node.grad.
    def clear_grad(self):
        self.node.clear_grad()


# Allows users set attributes of class Parameter within a module, which are automatically added to _params dict of module.
# This way, users can have untracked Variable attributes.
class Parameter(Variable):
    pass
